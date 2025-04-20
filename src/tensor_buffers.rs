use std::{error::Error, mem::size_of, sync::Mutex};

use bytemuck::Pod;
use bytes::BytesMut;
use tokio::sync::OnceCell;

use crate::{
    generated::tensor_buffers::{TensorBuffersMetadata, TensorMetadata},
    num_trait::Num,
    tensor_buffers_file::TensorBuffersFile,
    tensor_buffers_reader::{TensorBuffersRead, TensorBuffersReader},
    utils::hash_key,
    Tensor, TensorId,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// A struct to represent a collection of tensors stored in a memory-mapped file.
/// This struct provides methods to read tensor metadata and data from the file.
pub struct TensorBuffers<'a> {
    metadata_root: OnceCell<TensorBuffersMetadata<'a>>,
    reader: Mutex<TensorBuffersReader<TensorBuffersFile>>,
}

impl<'a> TensorBuffers<'a> {
    pub async fn open(url: &str) -> Result<Self> {
        let file = TensorBuffersFile::open(url).await?;
        let reader = TensorBuffersReader::new(file);
        Ok(TensorBuffers { metadata_root: OnceCell::new(), reader: Mutex::new(reader) })
    }

    async fn get_metadata_root(&self) -> Result<TensorBuffersMetadata<'a>> {
        if let Some(metadata_root) = self.metadata_root.get() {
            return Ok(*metadata_root);
        }
        let metadata_size = self.reader.lock().unwrap().get_metadata_size().await?;

        let mut buf = BytesMut::with_capacity(metadata_size);
        self.reader.lock().unwrap().read_metadata(&mut buf).await?;

        // Clone the buffer into a Box<[u8]> to extend its lifetime
        let owned_buf: Box<[u8]> = buf.to_vec().into_boxed_slice();
        // Leak the boxed buffer to get a 'a reference
        let leaked_buf: &'a [u8] = Box::leak(owned_buf);
        let metadata_root = flatbuffers::root::<TensorBuffersMetadata>(leaked_buf)
            .map_err(|_| "Failed to read metadata from mmap")?;
        self.metadata_root.set(metadata_root).map_err(|_| {
            "Failed to set metadata root. This should not happen if the metadata is read correctly."
        })?;
        Ok(*self.metadata_root.get().unwrap())
    }

    pub async fn get_tensor_metadata(&self, tensor_id: TensorId) -> Result<TensorMetadata> {
        let metadata_root = self.get_metadata_root().await?;
        let tensors = metadata_root.tensors().ok_or("No tensors found")?;
        let result = tensors
            .lookup_by_key(tensor_id, |field, key| field.key_compare_with_value(*key))
            .ok_or("Tensor ID not found in metadata")?;
        Ok(result)
    }

    pub async fn get_tensor_data_by_name<T>(&self, tensor_name: &str) -> Result<Tensor<T>>
    where
        T: Pod + Num,
    {
        let tensor_id = hash_key(tensor_name);
        self.get_tensor_data_by_id(tensor_id).await
    }

    pub async fn get_tensor_data_by_id<T>(&self, tensor_id: TensorId) -> Result<Tensor<T>>
    where
        T: Pod + Num,
    {
        let tensor_metadata = self.get_tensor_metadata(tensor_id).await?;
        let data_type = tensor_metadata.data_type();

        if data_type != T::data_type().into() {
            return Err(format!(
                "Tensor data type mismatch: expected {:?}, found {:?}",
                T::data_type(),
                data_type
            )
            .into());
        }

        let offset = tensor_metadata.data_offset() as usize;
        let size = tensor_metadata.data_size() as usize;

        if offset.checked_add(size).is_none() {
            return Err(
                format!("Tensor data range [{}, {}) out of bounds", offset, offset + size,).into(),
            );
        }

        let mut data_buf = BytesMut::with_capacity(size);
        self.reader.lock().unwrap().read_data_with_metadata(tensor_metadata, &mut data_buf).await?;

        if size % size_of::<T>() != 0 {
            return Err(format!(
                "Tensor data size ({}) is not a multiple of type size ({})",
                size,
                size_of::<T>()
            )
            .into());
        }
        let owned_buf: Box<[u8]> = data_buf.to_vec().into_boxed_slice();
        let leaked_buf: &'a [u8] = Box::leak(owned_buf);
        let data = bytemuck::cast_slice::<u8, T>(leaked_buf);
        let name = tensor_metadata.name();
        let shape = tensor_metadata
            .shape()
            .ok_or("Failed to get tensor shape from metadata")?
            .iter()
            .map(|dim| dim as usize)
            .collect::<Vec<_>>();
        Ok(Tensor::new(name, data, shape))
    }
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;
    use tokio::fs::File;

    use super::*;
    use crate::{
        generated::tensor_buffers::TensorBuffersMetadata,
        tensor_buffers_writer::TensorBuffersWrite, Tensor, TensorBuffersWriter,
    };

    #[tokio::test]
    async fn test_tensor_buffers_reader() {
        // Create a test tensor.
        let tensor = Tensor::new("1", &[1.0f32, 2.0, 3.0], vec![3]);
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_owned();
        let mut file = File::create(&path).await.unwrap();

        // Write the tensor to the buffer.
        let mut writer = TensorBuffersWriter::new(&mut file);
        writer.write_tensors(vec![tensor]).await.unwrap();

        // Reset file position for reading.
        let file = File::open(&path).await.unwrap();

        // Create a reader for the file.
        let mut reader = TensorBuffersReader::new(file);

        // Read the metadata section.
        let metadata_size = reader.get_metadata_size().await.unwrap();
        let mut metadata_buf = vec![0; metadata_size];
        reader.read_metadata(&mut metadata_buf).await.unwrap();
        // Parse the metadata and check tensor count.
        let tensor_buffers_metadata =
            flatbuffers::root::<TensorBuffersMetadata>(&metadata_buf).unwrap();
        assert_eq!(tensor_buffers_metadata.tensors().unwrap().len(), 1);
        // Read the tensor data using the metadata.
        let tensor_metadata = tensor_buffers_metadata.tensors().unwrap().get(0);
        let mut tensor_buf = vec![0; tensor_metadata.data_size() as usize];
        reader.read_data_with_metadata(tensor_metadata, &mut tensor_buf).await.unwrap();
        assert_eq!(tensor_buf.len(), 3 * std::mem::size_of::<f32>());
        let tensor_data = bytemuck::cast_slice::<u8, f32>(&tensor_buf);
        assert_eq!(tensor_data, &[1.0f32, 2.0, 3.0]);
    }
}
