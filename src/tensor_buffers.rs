use std::{error::Error, io::Read, mem::size_of, os::fd::AsRawFd, result::Result};

use memmap2::Mmap;

use crate::{
    generated::tensor_buffers::{TensorBuffersMetadata, TensorMetadata},
    num_trait::Num,
    types::TensorId,
    Tensor,
};

/// A struct to represent a collection of tensors stored in a memory-mapped file.
/// This struct provides methods to read tensor metadata and data from the file.
pub struct TensorBuffers {
    mmap: Mmap,
}

impl TensorBuffers {
    pub fn from_read<R>(reader: &R) -> Result<Self, Box<dyn Error>>
    where
        R: Read + AsRawFd,
    {
        let mmap = unsafe { Mmap::map(reader)? };
        if mmap.len() == 0 {
            return Err("File is empty".into());
        }
        Ok(TensorBuffers { mmap })
    }

    fn get_metadata_root(&self) -> Result<TensorBuffersMetadata, Box<dyn Error>> {
        // Flatbuffers metadata is at the end of the file.
        // The last 4 bytes of the file contain the root table offset (little endian u32).
        if self.mmap.len() < 4 {
            return Err("File too small to contain metadata root offset".into());
        }
        let len = self.mmap.len();
        let offset_bytes = &self.mmap[len - 4..];
        let root_offset = u32::from_le_bytes(offset_bytes.try_into().unwrap()) as usize;
        if root_offset > len - 4 {
            return Err("Metadata root offset out of bounds".into());
        }
        let metadata_root =
            flatbuffers::root::<TensorBuffersMetadata>(&self.mmap[root_offset..len - 4])
                .map_err(|_| "Failed to read metadata from end of mmap")?;
        Ok(metadata_root)
    }

    pub fn get_tensor_metadata(
        &self,
        tensor_id: TensorId,
    ) -> Result<TensorMetadata, Box<dyn Error>> {
        let metadata_root = self.get_metadata_root()?;
        let tensors = metadata_root.tensors().ok_or("No tensors found")?;
        if tensor_id >= tensors.len() {
            println!("Tensor ID {} out of bounds (max {})", tensor_id, tensors.len());
            return Err(
                format!("Tensor ID {} out of bounds (max {})", tensor_id, tensors.len()).into()
            );
        }

        Ok(tensors.get(tensor_id))
    }

    pub fn get_tensor_data<T>(&self, tensor_id: TensorId) -> Result<Tensor<T>, Box<dyn Error>>
    where
        T: bytemuck::Pod + Num,
    {
        let tensor_metadata = self.get_tensor_metadata(tensor_id)?;
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

        // Check if the requested range is within the bounds of the mmap
        if offset.checked_add(size).is_none() || offset + size > self.mmap.len() {
            return Err(format!(
                "Tensor data range [{}, {}) out of bounds of the file (len {})",
                offset,
                offset + size,
                self.mmap.len()
            )
            .into());
        }
        let tensor_data_bytes = &self.mmap[offset..(offset + size)];

        if size % size_of::<T>() != 0 {
            return Err(format!(
                "Tensor data size ({}) is not a multiple of the type size ({})",
                size,
                size_of::<T>()
            )
            .into());
        }
        let typed_slice: &[T] = bytemuck::cast_slice::<u8, T>(tensor_data_bytes);
        let id = tensor_id;
        let data = typed_slice;
        let shape = tensor_metadata
            .shape()
            .ok_or("Failed to get tensor shape from metadata")
            .map(|s| s.iter().map(|dim| dim as usize).collect::<Vec<usize>>())?;

        Ok(Tensor::new(id, data, shape))
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempfile;

    use super::*;
    use crate::tensor_buffers_writer::TensorBuffersWriter;

    #[test]
    fn test_tensor_buffers() {
        let tensor1 = Tensor::new(0, &[1.0f32, 2.0, 3.0], vec![3]);
        let tensor2 = Tensor::new(1, &[4.0f32, 5.0, 6.0], vec![3]);
        let tensors = vec![tensor1, tensor2];

        // Write tensors to a file
        let mut file = tempfile().unwrap();
        TensorBuffersWriter::write(&mut file, tensors).unwrap();

        // Read the file back
        let tensor_buffers = TensorBuffers::from_read(&file).unwrap();

        // Read the tensors back
        let read_tensor1: Tensor<f32> = tensor_buffers.get_tensor_data(0).unwrap();
        let read_tensor2: Tensor<f32> = tensor_buffers.get_tensor_data(1).unwrap();

        assert_eq!(read_tensor1.data(), &[1.0f32, 2.0, 3.0]);
        assert_eq!(read_tensor2.data(), &[4.0f32, 5.0, 6.0]);
    }
}
