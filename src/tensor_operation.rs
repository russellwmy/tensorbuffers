use flatbuffers::{FlatBufferBuilder, WIPOffset};

use crate::{
    generated::tensor_buffers::{Operation, OperationMetadata, OperationMetadataArgs},
    TensorId, TensorOperationId,
};

#[derive(Debug, Clone)]
pub struct TensorOperation {
    id: TensorOperationId,
    operation: Operation,
    input_operations: Vec<TensorOperationId>,
    output: TensorId,
}

impl TensorOperation {
    pub fn new(
        id: TensorOperationId,
        operation: Operation,
        input_operations: Vec<TensorOperationId>,
        output: TensorId,
    ) -> Self {
        TensorOperation { id, operation, input_operations, output }
    }

    pub fn id(&self) -> TensorOperationId {
        self.id
    }

    pub fn operation(&self) -> &Operation {
        &self.operation
    }

    pub fn input_operations(&self) -> &[TensorOperationId] {
        &self.input_operations
    }

    pub fn output(&self) -> &TensorId {
        &self.output
    }
}

impl TensorOperation {
    pub fn with_metadata(metadata: &OperationMetadata) -> Self {
        let id = metadata.id();
        let operation = metadata.operation();
        let input_operations = match metadata.input_operations() {
            Some(inputs) => inputs.iter().map(|x| x.to_owned()).collect(),
            None => Vec::new(),
        };
        let output = metadata.output();
        TensorOperation::new(id, operation, input_operations, output)
    }

    pub fn build_table<'a>(
        builder: &mut FlatBufferBuilder<'a>,
        tensor_operation: TensorOperation,
    ) -> WIPOffset<OperationMetadata<'a>> {
        let operation = *tensor_operation.operation();
        let output = *tensor_operation.output();
        let input_operations = builder.create_vector(&tensor_operation.input_operations);
        OperationMetadata::create(builder, &OperationMetadataArgs {
            id: tensor_operation.id(),
            operation: operation,
            input_operations: Some(input_operations),
            output: output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Operation;

    #[test]
    fn test_tensor_operation() {
        let id = 1;
        let operation = Operation::Add;
        let input_operations = vec![2, 3];
        let output = 4;

        let tensor_operation =
            TensorOperation::new(id, operation, input_operations.clone(), output);

        assert_eq!(tensor_operation.id(), id);
        assert_eq!(tensor_operation.operation(), &operation);
        assert_eq!(tensor_operation.input_operations(), &input_operations);
        assert_eq!(tensor_operation.output(), &output);
    }
}
