use std::io::Result;

use bytemuck::Pod;
use flatbuffers::FlatBufferBuilder;
use tokio::io::{AsyncSeek, AsyncWrite, AsyncWriteExt};

use crate::{constants::MAGIC_BYTES, Num, Tensor, TensorBuffers, TensorOperation};

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
    async fn write<'a, T>(
        &mut self,
        tensors: Vec<Tensor<'a, T>>,
        operations: Vec<TensorOperation>,
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
    async fn write<'a, T>(
        &mut self,
        tensors: Vec<Tensor<'a, T>>,
        operations: Vec<TensorOperation>,
    ) -> Result<()>
    where
        T: Pod + Num,
    {
        // Write the initial magic bytes to identify the file format.
        self.writer.write_all(MAGIC_BYTES).await?;

        // Track the starting offset for each tensor's data.
        let mut data_offsets = Vec::with_capacity(tensors.len());
        // Offset starts after the magic bytes.
        let mut current_offset = 4u64;

        // Write each tensor's data and record its offset.
        for t in tensors.iter() {
            data_offsets.push(current_offset as u32);
            // Convert tensor data to bytes.
            let data_bytes = bytemuck::cast_slice::<T, u8>(t.data());
            let data_size = data_bytes.len() as u32;
            self.writer.write_all(data_bytes).await?;
            current_offset += data_size as u64;
        }

        let mut builder = FlatBufferBuilder::new();

        // Build FlatBuffers metadata for all tensors.
        let mut tensor_metadata_offsets = Vec::with_capacity(tensors.len());

        for (i, t) in tensors.iter().enumerate() {
            // Create FlatBuffers metadata for this tensor.
            let tensor_metadata = Tensor::build_table(&mut builder, &t, data_offsets[i] as usize);
            tensor_metadata_offsets.push(tensor_metadata);
        }

        let mut operations_metadata_offsets = Vec::with_capacity(operations.len());
        // Write the operations to the writer.
        for op in operations {
            // Serialize and write the operation.
            let operation_metadata = TensorOperation::build_table(&mut builder, op);
            operations_metadata_offsets.push(operation_metadata);
        }

        let tensor_buffers_metadata = TensorBuffers::build_table(
            &mut builder,
            &tensor_metadata_offsets,
            &operations_metadata_offsets,
        );
        builder.finish(tensor_buffers_metadata, None);

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
        writer.write(vec![tensor1, tensor2], vec![]).await.unwrap();

        // Seek to the start and verify file is not empty.
        file.seek(SeekFrom::Start(0)).await.unwrap();
        assert!(file.metadata().await.unwrap().len() > 0);
    }
}
