# TensorBuffers

TensorBuffers is a data storage format optimized for efficient partial reads from large tensor collections. It is designed to:

- Support reading tensor collections larger than available RAM
- Leverage [HTTP Range Requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests) for efficient remote access from storage services like AWS S3 or Google Cloud Storage

## File Format

```markdown
|-------------------------------------------------|
| Tensor Data (in order of metadata)              |
|-------------------------------------------------|
| Tensor Buffers Metadata (Flatbuffers)           |
|-------------------------------------------------|
| Root Table Offset (4 bytes)                     |
|-------------------------------------------------|
```
