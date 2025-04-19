use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Seek, Write},
};

use bytemuck::Pod;
use flatbuffers::FlatBufferBuilder;

use crate::{
    generated::tensor_buffers::{
        TensorBuffersMetadata, TensorBuffersMetadataArgs, TensorMetadata, TensorMetadataArgs,
    },
    Num, Tensor, VERSION,
};

pub struct TensorBuffersWriter;

impl TensorBuffersWriter {
    pub fn write<'a, T, W>(
        writer: &mut W,
        tensors: impl IntoIterator<Item = Tensor<'a, T>>,
    ) -> Result<(), Box<dyn Error>>
    where
        T: Pod + Num,
        W: Write + Seek, // Need Seek to get the current position for offsets
    {
        // Collect tensors into a Vec to allow multiple iterations
        let tensor_vec: Vec<_> = tensors.into_iter().collect();

        // Calculate offsets for each tensor's data
        let mut data_offsets = Vec::with_capacity(tensor_vec.len());
        let mut current_offset = 0u64;

        // Write tensor data sequentially
        for t in tensor_vec.iter() {
            data_offsets.push(current_offset as u32);
            let data_bytes = bytemuck::cast_slice::<T, u8>(t.data());
            let data_size = data_bytes.len() as u32;
            writer.write_all(data_bytes)?;
            current_offset += data_size as u64;
        }

        // Build metadata with correct offsets
        let mut builder = FlatBufferBuilder::new();
        let mut tensor_metadata_offsets = Vec::with_capacity(tensor_vec.len());
        for (i, t) in tensor_vec.iter().enumerate() {
            let shape = t.shape().iter().map(|&dim| dim as u32).collect::<Vec<u32>>();
            let shape_offset = builder.create_vector::<u32>(&shape);
            let data_type = t.data_type();
            let data_bytes = bytemuck::cast_slice::<T, u8>(t.data());
            let tensor_metadata = TensorMetadata::create(&mut builder, &TensorMetadataArgs {
                id: t.id() as u32,
                data_type: data_type.into(),
                data_offset: data_offsets[i],
                data_size: data_bytes.len() as u32,
                shape: Some(shape_offset),
                ..Default::default()
            });
            tensor_metadata_offsets.push(tensor_metadata);
        }

        let version_offset = builder.create_string(VERSION);
        let tensors_offset = builder.create_vector(&tensor_metadata_offsets);
        let tensor_store_metadata =
            TensorBuffersMetadata::create(&mut builder, &TensorBuffersMetadataArgs {
                version: Some(version_offset),
                tensors: Some(tensors_offset),
                ..Default::default()
            });
        builder.finish(tensor_store_metadata, None);

        // Write metadata after tensor data
        let flatbuffer_data = builder.finished_data();
        writer.write_all(flatbuffer_data)?;
        writer.write((current_offset as u32).to_le_bytes().as_ref())?;

        writer.flush()?;
        Ok(())
    }

    pub fn write_to_file<'a, T>(
        path: &str,
        tensors: impl IntoIterator<Item = Tensor<'a, T>>,
    ) -> Result<(), Box<dyn Error>>
    where
        T: Pod + Num,
    {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        Self::write(&mut writer, tensors)?;
        writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::SeekFrom;

    use tempfile::tempfile;

    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_write_tensor_buffers() {
        let tensor1 = Tensor::new(1, &[1.0f32, 2.0, 3.0], vec![3]);
        let tensor2 = Tensor::new(2, &[4.0f32, 5.0, 6.0], vec![3]);

        let mut file = tempfile().unwrap();

        TensorBuffersWriter::write(&mut file, vec![tensor1, tensor2]).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        assert!(file.metadata().unwrap().len() > 0);
    }
}
