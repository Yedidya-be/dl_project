#!/bin/bash

for file in *.npy; do
    new_name=${file%.0}
    mv "$file" "$new_name"
done

