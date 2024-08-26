# FOTOSOP 

Jadi ini program untuk fotosop 

## Requirement 
Pastikan anda menggunakan linux, jangan Windows (JELEK, saya config windows ga selese-selese [mungkin saya yang skill issue]). 
Linux yang saya gunakan sendiri adalah Ubuntu 24.04
Pastikan lagi 
- sudah menginstall CUDA Toolkit [tutorial](https://developer.nvidia.com/cuda-downloads?target_os=Linux)
- sudah menginstall opencv yang dibuild dengan CUDA [tutorial](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html?ref=wasyresearch.com)
- Pastikan menginstall Cmake juga dengan perintah berikut (kalo gasalah)

```bash
sudo apt install cmake 
```

Note: Saya gatau caranya di-windows jujur karena ya susah atau saya memang skill issue, jadi tolong dicek di Linux atau kalau mungkin bisa coba saja WSL.

## Build 
jalankan perintah ini 
```bash
chmod +x ./run.sh
```
lalu
```bash
./run.sh
```

atau bisa jalankan secara manual saja dengan command berikut
```bash
rm -rf ./build
mkdir build
cd build
cmake ../
make
mv ./app ../
cd ../
```

NOTE: saya gak tau cara build pake makefile langsung, tutorialnya susah banget. Kebanyakan tutorial ngasihnya pake cmake. 

## Run 
setelah proses build akan muncul file ***app***
jalankan file tersebut dengan command 

```bash
./app /path/to/file.txt type(1, 2, 3, 4, 5) optional_params

contoh: 

./app /path/to/file.txt 1

```

ada beberapa command yang memerlukan argumen tambahan untuk fungsi-fungsi lain selain greyscale dan Sobel (Edge Detecting)
argumen tambahannya: alpha dan sigma
```bash
Saturation: /path/to/file 1 alpha
Contrast: /path/to/file 2 alpha
Gaussian: /path/to/file 4 sigma
Sobe: /path/to/file 5 

```        


