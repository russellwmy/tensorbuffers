use std::{
    future::Future,
    io::{Error, ErrorKind, Result, SeekFrom},
    pin::Pin,
    task::{ready, Context, Poll},
};

use bytes::Bytes;
use tokio::{
    fs::File,
    io::{AsyncRead, AsyncSeek, ReadBuf},
};
use tracing::{debug, info};
enum ReadState {
    Idle,
    Fetch(Pin<Box<dyn Future<Output = Result<Bytes>> + Send>>),
}

pub struct RemoteFile {
    url: String,
    offset: u64,
    file_size: u64,
    state: ReadState,
}

impl RemoteFile {
    pub async fn open(url: &str) -> Result<Self> {
        let file_size = Self::fetch_file_size(url.to_string()).await?;

        Ok(RemoteFile { url: url.to_string(), file_size, offset: 0, state: ReadState::Idle })
    }
}

impl RemoteFile {
    // Fetches and caches the file size from the remote server using a HEAD request.
    async fn fetch_file_size(url: String) -> Result<u64> {
        let client = reqwest::Client::new();
        let response =
            client.head(url).send().await.map_err(|e| {
                Error::new(ErrorKind::Other, format!("Failed to send request: {}", e))
            })?;
        if response.status().is_success() {
            if let Some(content_length) = response.headers().get(reqwest::header::CONTENT_LENGTH) {
                debug!("Content-Length: {:?}", content_length);
                if let Ok(size) = content_length.to_str() {
                    let parsed_size = size.parse::<u64>().map_err(|_| {
                        Error::new(ErrorKind::InvalidData, "Invalid content length")
                    })?;
                    debug!("File size: {}", parsed_size);
                    return Ok(parsed_size);
                }
            }
        }
        Err(Error::new(ErrorKind::Other, "Failed to get file size"))
    }

    async fn fetch_range(url: String, offset: u64, size: u64) -> Result<Bytes> {
        let client = reqwest::Client::new();
        let range = format!("bytes={}-{}", offset, offset + size - 1);
        let response =
            client.get(url).header(reqwest::header::RANGE, range).send().await.map_err(|e| {
                Error::new(ErrorKind::Other, format!("Failed to send request: {}", e))
            })?;

        if response.status().is_success() {
            let bytes = response.bytes().await.map_err(|e| {
                Error::new(ErrorKind::Other, format!("Failed to read response: {}", e))
            })?;
            Ok(bytes)
        } else {
            Err(Error::new(ErrorKind::Other, "Failed to read remote file"))
        }
    }
}

impl AsyncRead for RemoteFile {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<Result<()>> {
        let this = self.get_mut();

        loop {
            match &mut this.state {
                ReadState::Idle => {
                    let size = (this.file_size - this.offset).min(buf.capacity() as u64);
                    if size == 0 {
                        return Poll::Ready(Ok(()));
                    }
                    debug!("Fetching {} bytes from offset {}", size, this.offset);
                    let fut = Box::pin(Self::fetch_range(this.url.clone(), this.offset, size));
                    this.state = ReadState::Fetch(fut);
                }
                ReadState::Fetch(fut) => {
                    let bytes = ready!(fut.as_mut().poll(cx))?;
                    debug!("Fetched {} bytes", bytes.len());
                    buf.put_slice(&bytes);
                    this.offset += bytes.len() as u64;
                    this.state = ReadState::Idle;
                    return Poll::Ready(Ok(()));
                }
            }
        }
    }
}

impl AsyncSeek for RemoteFile {
    fn start_seek(self: Pin<&mut Self>, position: SeekFrom) -> Result<()> {
        let this = self.get_mut();

        match position {
            SeekFrom::Start(pos) => this.offset = pos,
            SeekFrom::End(pos) => {
                let base = this.file_size;
                this.offset = base
                    .checked_add_signed(pos)
                    .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "Seek overflow"))?;
            }
            SeekFrom::Current(pos) => {
                this.offset = this
                    .offset
                    .checked_add_signed(pos)
                    .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "Seek overflow"))?;
            }
        }
        debug!("Seek to offset: {}", this.offset);
        Ok(())
    }

    fn poll_complete(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<u64>> {
        Poll::Ready(Ok(self.offset))
    }
}

impl Drop for RemoteFile {
    fn drop(&mut self) {
        info!("Dropping RemoteFile");
    }
}
pub enum TensorBuffersFile {
    Local(File),
    Remote(RemoteFile),
}

impl TensorBuffersFile {
    pub async fn open(url: &str) -> Result<Self> {
        if url.starts_with("file://") {
            let path = &url[7..];
            Ok(TensorBuffersFile::Local(File::open(path).await?))
        } else if url.starts_with("https://") {
            Ok(TensorBuffersFile::Remote(RemoteFile::open(url).await?))
        } else {
            Err(Error::new(ErrorKind::InvalidInput, "Unsupported URI scheme"))
        }
    }
}

impl AsyncRead for TensorBuffersFile {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<Result<()>> {
        match self.get_mut() {
            TensorBuffersFile::Local(file) => Pin::new(file).poll_read(cx, buf),
            TensorBuffersFile::Remote(remote) => Pin::new(remote).poll_read(cx, buf),
        }
    }
}

impl AsyncSeek for TensorBuffersFile {
    fn start_seek(self: Pin<&mut Self>, position: SeekFrom) -> Result<()> {
        match self.get_mut() {
            TensorBuffersFile::Local(file) => Pin::new(file).start_seek(position),
            TensorBuffersFile::Remote(remote) => Pin::new(remote).start_seek(position),
        }
    }

    fn poll_complete(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<u64>> {
        match self.get_mut() {
            TensorBuffersFile::Local(file) => Pin::new(file).poll_complete(cx),
            TensorBuffersFile::Remote(remote) => Pin::new(remote).poll_complete(cx),
        }
    }
}

#[cfg(test)]
mod tests {
    use bytes::BytesMut;
    use tokio::io::{AsyncReadExt, AsyncSeekExt};

    use super::*;

    #[tokio::test]
    async fn test_remote_file() {
        let mut remote_file = RemoteFile::open(
            "https://raw.githubusercontent.com/russellwmy/tensorbuffers/refs/heads/main/LICENSE",
        )
        .await
        .unwrap();
        let mut buf = BytesMut::new();
        buf.resize(1024, 0);
        remote_file.read_exact(&mut buf).await.unwrap();
        assert!(buf.len() == 1024);

        let mut buf = BytesMut::new();
        buf.resize(1024, 0);
        remote_file.seek(SeekFrom::Start(1000)).await.unwrap();
        remote_file.read(&mut buf).await.unwrap();
        assert!(buf.len() == 1024);
    }
}
