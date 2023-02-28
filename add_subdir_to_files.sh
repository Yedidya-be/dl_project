#!/bin/bash

root_directory=$1

for sub_directory in $root_directory/*
do
  if [ -d "$sub_directory" ]; then
    for file in $sub_directory/*
    do
      mv "$file" "${sub_directory}/$(basename "$sub_directory")_$(basename "$file")"
    done
  fi
done

