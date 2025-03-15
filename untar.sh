#!/bin/bash

# Set input and output directories
RAW_DIR="raw_data"
DATA_DIR="data"

# Ensure the output directory exists
mkdir -p "$DATA_DIR"

# Find and extract all .tar files
find "$RAW_DIR" -type f -name "*.tar" | while read tarfile; do
    # Get relative path of tar file
    rel_path="${tarfile#$RAW_DIR/}" 
    # Remove .tar extension
    target_dir="${rel_path%.tar}"
    # Define the output path
    output_path="$DATA_DIR/"

    echo "Extracting $tarfile to $output_path"
    mkdir -p "$output_path"
    tar -xf "$tarfile" -C "$output_path"
done

echo "Extraction completed."

