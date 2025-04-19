use std::{path::Path, process::Command};

fn main() {
    use std::fs;

    let schema_file = "schema.fbs";
    let output_file = "src/schema.rs";

    // Ensure the output directory exists
    let output_dir = Path::new("src");
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir).expect("Failed to create output directory");
    }

    // Run the flatc command to generate Rust bindings
    let status = Command::new("flatc")
        .args(&["--rust", "--filename-suffix", "", "--gen-onefile", "-o", "src"])
        .arg(schema_file) // Changed from .args(&schema_file) to .arg(schema_file)
        .status()
        .expect("Failed to execute flatc");

    if !status.success() {
        panic!("flatc failed to generate bindings");
    }
    // Rename the generated file to "generated.rs"
    let generated_file = "src/generated.rs";
    fs::rename(output_file, generated_file).expect("Failed to rename binding.rs to generated.rs");

    println!("cargo:rerun-if-changed={}", schema_file);
    println!("Bindings generated at {}", generated_file);
}
