use std::{mem::size_of, sync::Mutex};

use bytemuck::Pod;
use bytes::BytesMut;
use flatbuffers::{FlatBufferBuilder, WIPOffset};
use tokio::sync::OnceCell;

use crate::{
    constants::VERSION,
    generated::tensor_buffers::{
        OperationMetadata, TensorBuffersMetadata, TensorBuffersMetadataArgs, TensorMetadata,
    },
    num_trait::Num,
    tensor_buffers_file::TensorBuffersFile,
    tensor_buffers_reader::{TensorBuffersRead, TensorBuffersReader},
    tensor_operation,
    utils::hash_key,
    Result, Tensor, TensorId, TensorOperation, TensorOperationId,
};
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

    pub async fn get_tensor_operation_by_id(
        &self,
        operation_id: TensorOperationId,
    ) -> Result<TensorOperation> {
        let metadata_root = self.get_metadata_root().await?;
        let operations = metadata_root.operations().ok_or("No operations found")?;
        let result = operations
            .lookup_by_key(operation_id, |field, key| field.key_compare_with_value(*key))
            .ok_or("Operation ID not found in metadata")?;
        Ok(TensorOperation::with_metadata(&result))
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

        let mut buf = BytesMut::new();
        buf.resize(size, 0);
        self.reader.lock().unwrap().read_data_with_metadata(tensor_metadata, &mut buf).await?;

        if size % size_of::<T>() != 0 {
            return Err(format!(
                "Tensor data size ({}) is not a multiple of type size ({})",
                size,
                size_of::<T>()
            )
            .into());
        }

        Tensor::new_with_metadata_and_data(tensor_metadata, buf.to_vec())
    }
}

impl<'a> TensorBuffers<'a> {
    pub fn build_table(
        builder: &mut FlatBufferBuilder<'a>,
        tensor_metadata_offsets: &[WIPOffset<TensorMetadata<'a>>],
        tensor_operation_offsets: &[WIPOffset<OperationMetadata<'a>>],
    ) -> WIPOffset<TensorBuffersMetadata<'a>> {
        // Create FlatBuffers metadata for the file.
        let version_offset = builder.create_string(VERSION);
        let tensors_offset = builder.create_vector(&tensor_metadata_offsets);
        let operations_offset = builder.create_vector(&tensor_operation_offsets);
        TensorBuffersMetadata::create(builder, &TensorBuffersMetadataArgs {
            version: Some(version_offset),
            tensors: Some(tensors_offset),
            operations: Some(operations_offset),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use bytemuck::cast_slice;
    use tempfile::NamedTempFile;
    use tokio::fs::File;

    use super::*;
    use crate::{
        generated::tensor_buffers::TensorBuffersMetadata,
        tensor_buffers_writer::TensorBuffersWrite, Operation, Tensor, TensorBuffersWriter,
    };

    #[tokio::test]
    async fn test_tensor_buffers_reader() {
        // Create a test tensor.
        let tensor_1 = Tensor::new("1", &[1.0f32, 2.0, 3.0], vec![3]);
        let tensor_2 = Tensor::new("2", &[1.0f32, 2.0, 3.0], vec![3]);
        let tensor_operation1 = TensorOperation::new(1, Operation::None, vec![], tensor_1.id());
        let tensor_operation2 =
            TensorOperation::new(2, Operation::Add, vec![tensor_operation1.id()], tensor_2.id());
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_owned();
        let mut file = File::create(&path).await.unwrap();

        // Write the tensor operation to the buffer.
        let mut writer = TensorBuffersWriter::new(&mut file);
        writer.write(vec![tensor_1], vec![tensor_operation1, tensor_operation2]).await.unwrap();

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
        let tensor_data = cast_slice::<u8, f32>(&tensor_buf);
        assert_eq!(tensor_data, &[1.0f32, 2.0, 3.0]);

        // Read the tensor operation data using the metadata.
        let tensor_operation_metadata = tensor_buffers_metadata.operations().unwrap().get(0);
        assert!(tensor_operation_metadata.id() == 1);
        assert!(tensor_operation_metadata.operation() == Operation::None);
        assert!(tensor_operation_metadata.input_operations().unwrap().len() == 0);
    }
}
