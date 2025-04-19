const VERSION: &str = "1.0.0";

mod generated;
mod num_trait;
mod tensor;
mod tensor_buffers;
mod tensor_buffers_writer;

pub mod types;
pub use num_trait::{DataType, Num, Zero};
pub use tensor::Tensor;
pub use tensor_buffers::TensorBuffers;
pub use tensor_buffers_writer::TensorBuffersWriter;
