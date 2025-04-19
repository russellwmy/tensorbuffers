use std::fmt::Debug;

use crate::num_trait::{DataType, Num};

#[derive(Debug, Clone)]
pub struct Tensor<'a, T> {
    id: usize,
    data: &'a [T],
    data_type: DataType,
    shape: Vec<usize>,
}

impl<'a, T> Tensor<'a, T>
where
    T: Num + Debug,
{
    pub fn new(id: usize, data: &'a [T], shape: Vec<usize>) -> Self {
        let data_type = T::data_type();
        Tensor { id, data, data_type, shape }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn data(&self) -> &'a [T] {
        self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_new_and_getters() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let id = 1;
        let tensor = Tensor::new(id, &data, shape.clone());

        assert_eq!(tensor.id(), id);
        assert_eq!(tensor.data(), &data[..]);
        assert_eq!(tensor.shape(), &shape[..]);
        assert_eq!(tensor.data_type(), DataType::Float32);
    }

    #[test]
    fn test_tensor_new_different_type() {
        let data: Vec<i32> = vec![1, 2, 3, 4];
        let shape = vec![2, 2];
        let id = 2;
        let tensor = Tensor::new(id, &data, shape.clone());

        assert_eq!(tensor.id(), id);
        assert_eq!(tensor.data(), &data[..]);
        assert_eq!(tensor.shape(), &shape[..]);
        assert_eq!(tensor.data_type(), DataType::Int32);
    }

    #[test]
    fn test_tensor_empty_data() {
        let data: Vec<u8> = vec![];
        let shape = vec![0];
        let id = 3;
        let tensor = Tensor::new(id, &data, shape.clone());

        assert_eq!(tensor.id(), id);
        assert_eq!(tensor.data(), &data[..]);
        assert_eq!(tensor.shape(), &shape[..]);
        assert_eq!(tensor.data_type(), DataType::UInt8);
    }

    #[test]
    fn test_tensor_clone() {
        let data: Vec<f64> = vec![1.0, 2.0];
        let shape = vec![2];
        let id = 4;
        let tensor1 = Tensor::new(id, &data, shape.clone());
        let tensor2 = tensor1.clone();

        assert_eq!(tensor1.id(), tensor2.id());
        assert_eq!(tensor1.data(), tensor2.data());
        assert_eq!(tensor1.shape(), tensor2.shape());
        assert_eq!(tensor1.data_type(), tensor2.data_type());
        // Ensure it's a shallow clone for the data reference
        assert!(std::ptr::eq(tensor1.data(), tensor2.data()));
    }
}
