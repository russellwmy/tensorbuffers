use std::{error::Error, io::SeekFrom};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt};

use crate::{constants::MAGIC_BYTES, generated::tensor_buffers::TensorMetadata};

/// Trait for reading tensor data and metadata from an async source.
/// Allows for different implementations of how tensors are read.
pub trait TensorBuffersRead {
    /// Reads the size of the metadata section from the file.
    async fn get_metadata_size(&mut self) -> Result<usize, Box<dyn Error>>;

    /// Reads the metadata of the TensorBuffers file into the provided buffer.
    /// The metadata is expected to be at the end of the file, preceded by its size and magic bytes.
    ///
    /// # Arguments
    /// * `buf` - A mutable slice of `u8` to store the read metadata.
    ///
    /// # Returns
    /// Returns `Ok(())` on success, or a `Box<dyn Error>` on failure.
    async fn read_metadata(&mut self, buf: &mut [u8]) -> Result<(), Box<dyn Error>>;

    /// Reads the data for a specific tensor based on its metadata into the provided buffer.
    ///
    /// # Arguments
    /// * `metadata` - The `TensorMetadata` for the tensor to read.
    /// * `buf` - A mutable slice of `u8` to store the read tensor data.
    ///
    /// # Returns
    /// Returns `Ok(())` on success, or a `Box<dyn Error>` on failure.
    async fn read_data_with_metadata<'a>(
        &mut self,
        metadata: TensorMetadata<'a>,
        buf: &mut [u8],
    ) -> Result<(), Box<dyn Error>>;
}

/// Struct that implements `TensorBuffersRead` for any type that implements `AsyncRead` and `AsyncSeek`.
pub struct TensorBuffersReader<R>
where
    R: AsyncRead + AsyncSeek, // R must support async reading and seeking.
{
    reader: R, // The underlying async reader.
}

impl<'a, R> TensorBuffersReader<R>
where
    R: AsyncRead + AsyncSeek,
{
    /// Creates a new `TensorBuffersReader` from the given reader.
    ///
    /// # Arguments
    /// * `reader` - An object that implements `AsyncRead` and `AsyncSeek`.
    pub fn new(reader: R) -> Self {
        TensorBuffersReader { reader }
    }
}

impl<R> TensorBuffersRead for TensorBuffersReader<R>
where
    R: AsyncRead + AsyncSeek + Unpin,
{
    async fn get_metadata_size(&mut self) -> Result<usize, Box<dyn Error>> {
        // Seek to 8 bytes before the end of the file:
        // [metadata_size (4 bytes)][magic_bytes (4 bytes)] are at the end.
        self.reader.seek(SeekFrom::End(-8)).await?;

        // Read the 4 bytes representing the metadata size (little-endian u32).
        let mut metadata_size_buff = [0; 4];
        self.reader.read_exact(&mut metadata_size_buff).await?;

        let metadata_size = u32::from_le_bytes(metadata_size_buff) as usize;
        Ok(metadata_size)
    }

    /// Reads the metadata section from the file into `buf`.
    /// Assumes file layout: [tensor data][metadata][metadata_size][magic_bytes]
    async fn read_metadata(&mut self, buf: &mut [u8]) -> Result<(), Box<dyn Error>> {
        // Seek to the start to verify the initial magic bytes.
        self.reader.seek(SeekFrom::Start(0)).await?;
        let mut magic_buf = [0; 4];
        self.reader.read_exact(&mut magic_buf).await?;
        if magic_buf != MAGIC_BYTES {
            return Err("Invalid magic bytes".into());
        }

        // Get the size of the metadata section.
        let metadata_size = self.get_metadata_size().await?;

        // Ensure the provided buffer is large enough.
        if buf.len() < metadata_size {
            return Err("Buffer size is insufficient".into());
        }

        // Seek to the start of the metadata section:
        // It's located (metadata_size + 8) bytes before the end.
        self.reader.seek(SeekFrom::End(-(metadata_size as i64 + 8))).await?;

        // Read the metadata into the buffer.
        self.reader.read_exact(buf).await?;

        Ok(())
    }

    /// Reads the raw tensor data for a specific tensor into `buf`.
    /// Uses the offset and size from the provided `TensorMetadata`.
    async fn read_data_with_metadata<'a>(
        &mut self,
        tensor_metadata: TensorMetadata<'a>,
        buf: &mut [u8],
    ) -> Result<(), Box<dyn Error>> {
        let offset = tensor_metadata.data_offset() as u64;
        let size = tensor_metadata.data_size() as usize;

        // Seek to the tensor's data offset.
        self.reader.seek(SeekFrom::Start(offset)).await?;

        // Ensure the buffer is large enough for the tensor data.
        if buf.len() < size {
            return Err("Buffer size is insufficient".into());
        }

        // Read the tensor data into the buffer.
        self.reader.read_exact(buf).await?;

        // The caller is responsible for interpreting the buffer contents.
        Ok(())
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
    }
}
