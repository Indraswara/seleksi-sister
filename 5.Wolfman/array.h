#ifndef ARRAY_H
#define ARRAY_H

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream> 

using namespace std; 

template <class T>
class D_array {
private: 
    T* start_; 
    T* end_;

    void allocate(size_t size){
        cudaError_t result = cudaMalloc(&start_, size * sizeof(T));
        if (result != cudaSuccess) {
            start_ = end_ = 0;
            throw std::runtime_error("CUDA error in allocate: " + std::string(cudaGetErrorString(result)));
        }
        end_ = start_ + size;
    }

    void free(){
        if (start_ != 0) {
            cudaFree(start_);
            start_ = 0;
            end_ = 0;
        }
    }

public: 
    explicit D_array() : start_(0), end_(0) {}
    explicit D_array(size_t size) { allocate(size); }
    ~D_array() { free(); }
    void resize(size_t size) { free(); allocate(size); }
    size_t getSize() const { return end_ - start_; }
    const T* getData() const { return start_; }
    T* getData() { return start_; }

    void set(const T* src, size_t size){
        size_t minn = min(size, getSize());
        cudaError_t result = cudaMemcpy(start_, src, minn * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess) {
            throw std::runtime_error("CUDA error in set: " + std::string(cudaGetErrorString(result)));
        }
    }

    void get(T* dest, size_t size){
        size_t minn = min(size, getSize());
        cudaError_t result = cudaMemcpy(dest, start_, minn * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            throw std::runtime_error("CUDA error in get: " + std::string(cudaGetErrorString(result)));
        }
    }
};

#endif