# Wolfman 

## Running Program 

untuk masing-masing bahasa berikut 
### Run CUDA application

compile 
```sh 
nvcc -o app kernel.cu main.cu
```
```bash
app "$input_file" > "output/${base_name}CUDA.txt"
```

### Run Python script
```sh
cd python
```
```sh
python3 main.py "$input_file" > "output/${base_name}Py.txt"
```

### Compile and run Java application
```sh
cd java
```
```sh
java Main.java "$input_file" > "output/${base_name}Java.txt"
```

### Compile OpenCL 
pindah ke folder openCL
```sh
cd OpenCL
```
compile 
```sh
gcc -o kernel main.c -lOpenCL
```

running
```
./kernel "../$input_file" > "../output/${base_name}openCL.txt"
```

### openMP 
pindah dulu ke directory
```sh
cd OpenMP
```
compile
```sh
g++ -o app main.cpp -fopenmp
```
running
```sh
./app "../$input_file" > "../output/${base_name}openMP.txt"
```

### mau brutal dikit? 
#### buat di windows 
gabuat karena ribut...
#### buat di linux / WSL
jalanin run.sh 
```sh
chmod +x ./run.sh
```
```sh
./run.sh <input_file.txt> 
```
## Hasil Pengujian
Hasil pengujian terdapat pada folder output. Setiap testcase yang ada akan memilik output sesuai dengan nama input

Contoh:
input 
```sh
32.txt
```
maka outputnya
```
32{bahasa_pemrograman}.txt
```

## Unexpected Event 
folder input dan output terlalu besar keknya saya taruh di dalam link yang terpisah aja nanti dari sini. Ehe. 