#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input_file=$1
base_name=$(basename "$input_file" .txt)

# Run CUDA application
CUDA/app "$input_file" > "output/${base_name}CUDA.txt"

# Run Python script
python3 Python/main.py "$input_file" > "output/${base_name}Py.txt"

# Compile and run Java application
java Java/Main.java "$input_file" > "output/${base_name}Java.txt"

# Compile OpenCL 
cd OpenCL
gcc -o kernel main.c -lOpenCL
./kernel "../$input_file" > "../output/${base_name}openCL.txt"
cd ..

## openMP 
cd OpenMP
g++ -o app main.cpp -fopenmp
./app "../$input_file" > "../output/${base_name}openMP.txt"
cd ..

echo "Outputs have been saved to output/${base_name}CUDA.txt, output/${base_name}Py.txt, and output/${base_name}Java.txt"