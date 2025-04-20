# TensorBuffers Data Format

## File Format

```

+-----------------------------------------------+-------------------------------------------------------+
| Section                                       | Description                                           |
+-----------------------------------------------+-------------------------------------------------------+
| TensorBuffers Magic Bytes (4 B)               | File signature to identify the format                 |
| Tensor Data                                   | Raw tensor data stored sequentially                   |
| TensorBuffers Metadata (Flatbuffers)          | Metadata describing the tensors and file structure    |
| TensorBuffers Metadata Data Size (4 B)        | Size of the root table in the metadata section        |
| TensorBuffers Magic Bytes (4 B)               | File signature repeated at the end for validation     |
+-----------------------------------------------+-------------------------------------------------------+

```

## Data Model

### Supported Data Type

- Int8
- Int16
- Int32
- Int64
- UInt8
- UInt16
- UInt32
- UInt64
- Float32
- Float64

### TensorMetadata

```

+--------------+---------------------------------------------------+
| Field        | Description                                       |
+--------------+---------------------------------------------------+
| id           | Hash of the tensor's name for identification      |
| name         | Unique string identifier for the tensor           |
| shape        | Array of unsigned integers specifying dimensions  |
| data_type    | Type of data stored in the tensor                 |
| data_offset  | Byte offset to the tensor's data in the file      |
| data_size    | Number of bytes occupied by the tensor's data     |
+--------------+---------------------------------------------------+

```

### TensorBuffersMetadata

```

+-------------------+---------------------------------------------------------------+
| Field             | Description                                                   |
+-------------------+---------------------------------------------------------------+
| version           | Specifies the version of the TensorBuffers file format        |
| model             | Identifier or name of the associated machine learning model   |
| tensors           | Array of TensorMetadata objects for each tensor in the file   |
+-------------------+---------------------------------------------------------------+

```
