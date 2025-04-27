mod constants;
mod generated;
mod num_trait;
mod tensor;
mod tensor_buffers;
mod tensor_buffers_file;
mod tensor_buffers_reader;
mod tensor_buffers_writer;
mod utils;

pub use num_trait::{DataType, Num, Zero};
pub use tensor::Tensor;
pub use tensor_buffers::TensorBuffers;
pub use tensor_buffers_file::RemoteFile;
pub use tensor_buffers_writer::TensorBuffersWriter;

pub type TensorId = u64;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
