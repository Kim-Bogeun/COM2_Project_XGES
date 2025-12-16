# XGES Parallelization

### 소스 코드
- `src-cpp/BICScorerGPU.cu`: GPU Naive Kernel 구현
- `src-cpp/BICScorerGPU_Optimized.cu`: GPU Optimized Kernel 구현
- `src-cpp/BICScorerGPU_cuBLAS.cu`: GPU cuBLAS Library 구현
- `src-cpp/BICScorer.cpp`: CPU 버전(Baseline)
- `src-cpp/main.cpp`: 메인 프로그램

## 빌드

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 실행

### 1. CPU 버전
```bash
./build/src-cpp/xges --input <data.npy> (samples/Data_var200_samples600.npy)
```

### 2. GPU Custom Kernel
```bash
./build/src-cpp/xges --batch --input <data.npy>
```

### 3. GPU cuBLAS Library
```bash
./build/src-cpp/xges --batch --backend cublas --input <data.npy>
```

## 시스템 요구사항
- CUDA Toolkit 11.0+
- cuBLAS library
- CMake 3.18+
 

