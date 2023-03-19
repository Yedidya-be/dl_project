#!/bin/bash

# Create the train directory if it doesn't already exist
if [ ! -d "train" ]; then
    mkdir train
fi

# Loop through all subdirectories
for dir in */; do
    # Loop through all files in the subdirectory
    for file in "$dir"/*; do
        # Get the filename without the directory path
        filename=$(basename "$file")
        # Copy the file to the train directory with the subdirectory name as a prefix
        cp "$file" "train/${dir%/}_${filename}"
    done
done
