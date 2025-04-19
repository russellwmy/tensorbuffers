#!/bin/sh

# Install the Rust nightly toolchain with minimal profile (for advanced features)
rustup toolchain install nightly --profile=minimal

# Add the Rust formatter (rustfmt) for the nightly toolchain
rustup component add --toolchain nightly rustfmt 

# Install the FlatBuffers compiler (flatc) based on the platform
if [ "$(uname)" = "Darwin" ]; then
	# macOS
	curl -L https://github.com/google/flatbuffers/releases/download/v25.2.10/Mac.flatc.binary.zip -o flatc.zip
elif [ "$(uname)" = "Linux" ]; then
	# Linux
	curl -L https://github.com/google/flatbuffers/releases/download/v25.2.10/Linux.flatc.binary.clang++-18.zip -o flatc.zip
else
	echo "Unsupported platform: $(uname)"
	exit 1
fi

unzip flatc.zip
rm flatc.zip
chmod +x flatc
sudo mv flatc /usr/local/bin/flatc