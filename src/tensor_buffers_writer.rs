use std::io::Result;

use bytemuck::Pod;
use flatbuffers::FlatBufferBuilder;
use tokio::io::{AsyncSeek, AsyncWrite, AsyncWriteExt};

use crate::{
    constants::{MAGIC_BYTES, VERSION},
    generated::tensor_buffers::{
        TensorBuffersMetadata, TensorBuffersMetadataArgs, TensorMetadata, TensorMetadataArgs,
    },
    Num, Tensor,
};

// Define a trait for writing tensors to a destination.
// This trait abstracts the logic for serializing and writing tensors.
pub trait TensorBuffersWrite {
    /// Writes an iterator of tensors to the implementing writer.
    ///
    /// # Arguments
    /// * `tensors` - An iterator yielding `Tensor` objects.
    ///
    /// # Returns
    /// Returns `Ok(())` on success, or a `Box<dyn Error>` on failure.
    async fn write_tensors<'a, T>(
        &mut self,
        tensors: impl IntoIterator<Item = Tensor<'a, T>>,
    ) -> Result<()>
    where
        T: Pod + Num; // T must be plain old data and implement the custom Num trait.
}

// Implements `TensorBuffersWrite` for any type that implements AsyncWrite and AsyncSeek.
pub struct TensorBuffersWriter<W>
where
    W: AsyncWrite + AsyncSeek + Unpin, // W must support async writing and seeking.
{
    writer: W, // The underlying async writer.
}

impl<W> TensorBuffersWriter<W>
where
    W: AsyncWrite + AsyncSeek + Unpin,
{
    /// Creates a new `TensorBuffersWriter`.
    ///
    /// # Arguments
    /// * `writer` - An object that implements AsyncWrite and AsyncSeek.
    pub fn new(writer: W) -> Self {
        TensorBuffersWriter { writer }
    }
}

// Implements the serialization and writing logic for tensors.
impl<W> TensorBuffersWrite for TensorBuffersWriter<W>
where
    W: AsyncWrite + AsyncSeek + Unpin,
{
    /// Serializes and writes tensors to the underlying writer in a custom format.
    /// The format: magic bytes | tensor data | FlatBuffers metadata | metadata size | magic bytes.
    async fn write_tensors<'a, T>(
        &mut self,
        tensors: impl IntoIterator<Item = Tensor<'a, T>>,
    ) -> Result<()>
    where
        T: Pod + Num,
    {
        // Write the initial magic bytes to identify the file format.
        self.writer.write_all(MAGIC_BYTES).await?;

        // Collect tensors into a Vec for multiple passes (offsets, data, metadata).
        let tensor_vec: Vec<_> = tensors.into_iter().collect();

        // Track the starting offset for each tensor's data.
        let mut data_offsets = Vec::with_capacity(tensor_vec.len());
        // Offset starts after the magic bytes.
        let mut current_offset = 4u64;

        // Write each tensor's data and record its offset.
        for t in tensor_vec.iter() {
            data_offsets.push(current_offset as u32);
            // Convert tensor data to bytes.
            let data_bytes = bytemuck::cast_slice::<T, u8>(t.data());
            let data_size = data_bytes.len() as u32;
            self.writer.write_all(data_bytes).await?;
            current_offset += data_size as u64;
        }

        // Build FlatBuffers metadata for all tensors.
        let mut builder = FlatBufferBuilder::new();
        let mut tensor_metadata_offsets = Vec::with_capacity(tensor_vec.len());

        for (i, t) in tensor_vec.iter().enumerate() {
            // Convert shape to u32 for FlatBuffers.
            let shape = t.shape().iter().map(|&dim| dim as u32).collect::<Vec<u32>>();
            let shape_offset = builder.create_vector::<u32>(&shape);
            let data_type = t.data_type();
            let data_bytes = bytemuck::cast_slice::<T, u8>(t.data());
            let name = builder.create_string(t.name());

            // Create FlatBuffers metadata for this tensor.
            let tensor_metadata = TensorMetadata::create(&mut builder, &TensorMetadataArgs {
                id: t.id(),
                name: Some(name),
                data_type: data_type.into(),
                data_offset: data_offsets[i],
                data_size: data_bytes.len() as u32,
                shape: Some(shape_offset),
                ..Default::default()
            });
            tensor_metadata_offsets.push(tensor_metadata);
        }

        // Create FlatBuffers metadata for the file.
        let version_offset = builder.create_string(VERSION);
        let tensors_offset = builder.create_vector(&tensor_metadata_offsets);
        let tensor_store_metadata =
            TensorBuffersMetadata::create(&mut builder, &TensorBuffersMetadataArgs {
                version: Some(version_offset),
                tensors: Some(tensors_offset),
                ..Default::default()
            });
        builder.finish(tensor_store_metadata, None);

        // Write FlatBuffers metadata to the writer.
        let flatbuffer_data = builder.finished_data();
        self.writer.write_all(flatbuffer_data).await?;
        let metadata_size = flatbuffer_data.len() as u32;

        // Write the size of the metadata (little-endian u32).
        self.writer.write(metadata_size.to_le_bytes().as_ref()).await?;

        // Write trailing magic bytes to mark the end of the file.
        self.writer.write(MAGIC_BYTES).await?;
        self.writer.flush().await?;
        Ok(())
    }
}

// Test module.
#[cfg(test)]
mod tests {
    use std::io::SeekFrom;

    use tempfile::NamedTempFile;
    use tokio::{fs::File, io::AsyncSeekExt};

    use super::*;
    use crate::tensor::Tensor;

    // Test writing tensor buffers to a file.
    #[tokio::test]
    async fn test_write_tensor_buffers() {
        // Create two sample tensors.
        let tensor1 = Tensor::new("1", &[1.0f32, 2.0, 3.0], vec![3]);
        let tensor2 = Tensor::new("2", &[4.0f32, 5.0, 6.0], vec![3]);

        // Create a temporary file.
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_owned();
        let mut file = File::create(path).await.unwrap();

        // Write tensors to the file.
        let mut writer = TensorBuffersWriter::new(&mut file);
        writer.write_tensors(vec![tensor1, tensor2]).await.unwrap();

        // Seek to the start and verify file is not empty.
        file.seek(SeekFrom::Start(0)).await.unwrap();
        assert!(file.metadata().await.unwrap().len() > 0);
    }
}
