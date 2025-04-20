use std::fmt::Debug;

use tracing::debug;

use crate::{
    num_trait::{DataType, Num},
    utils::hash_key,
    TensorId,
};

#[derive(Debug, Clone)]
pub struct Tensor<'a, T> {
    id: TensorId,
    name: &'a str,
    data: &'a [T],
    data_type: DataType,
    shape: Vec<usize>,
}

impl<'a, T> Tensor<'a, T>
where
    T: Num + Debug,
{
    pub fn new(name: &'a str, data: &'a [T], shape: Vec<usize>) -> Self {
        let data_type = T::data_type();
        Tensor { id: hash_key(name), name, data, data_type, shape }
    }

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn name(&self) -> &'a str {
        self.name
    }

    pub fn data(&self) -> &[T] {
        self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }
}

impl<T> Drop for Tensor<'_, T> {
    fn drop(&mut self) {
        debug!("Dropping Tensor with name: {}", self.name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_new_and_getters() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let name = "input_1";
        let tensor = Tensor::new(name, &data, shape.clone());

        assert_eq!(tensor.id(), hash_key(name));
        assert_eq!(tensor.name(), name);
        assert_eq!(tensor.shape(), &shape[..]);
        assert_eq!(tensor.data_type(), DataType::Float32);
    }

    #[test]
    fn test_tensor_new_different_type() {
        let data: Vec<i32> = vec![1, 2, 3, 4];
        let shape = vec![2, 2];
        let name = "input_2";
        let tensor = Tensor::new(name, &data, shape.clone());

        assert_eq!(tensor.id(), hash_key(name));
        assert_eq!(tensor.name(), name);
        assert_eq!(tensor.shape(), &shape[..]);
        assert_eq!(tensor.data_type(), DataType::Int32);
    }

    #[test]
    fn test_tensor_empty_data() {
        let data: Vec<u8> = vec![];
        let shape = vec![0];
        let name = "input_3";
        let tensor = Tensor::new(name, &data, shape.clone());

        assert_eq!(tensor.id(), hash_key(name));
        assert_eq!(tensor.name(), name);
        assert_eq!(tensor.shape(), &shape[..]);
        assert_eq!(tensor.data_type(), DataType::UInt8);
    }

    #[test]
    fn test_tensor_clone() {
        let data: Vec<f64> = vec![1.0, 2.0];
        let shape = vec![2];
        let name = "input_4";
        let tensor1 = Tensor::new(name, &data, shape.clone());
        let tensor2 = tensor1.clone();

        assert_eq!(tensor1.id(), tensor2.id());
        assert_eq!(tensor1.data(), tensor2.data());
        assert_eq!(tensor1.shape(), tensor2.shape());
        assert_eq!(tensor1.data_type(), tensor2.data_type());
    }
}
