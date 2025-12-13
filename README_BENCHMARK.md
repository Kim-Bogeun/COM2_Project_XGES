# XGES Benchmark Package

## 구성 파일

### 소스 코드
- `src-cpp/BICScorerGPU_Optimized.cu`: GPU Custom Kernel 구현
- `src-cpp/BICScorerGPU_cuBLAS.cu`: GPU cuBLAS Library 구현
- `src-cpp/BICScorer.cpp`: CPU 구현
- `src-cpp/main.cpp`: 메인 프로그램

### 데이터 생성
- `generate_data_and_ground_truth.py`: Scale-free DAG 데이터 생성

### 샘플 데이터셋
- `samples/Data_var200_samples600.npy`: 200 variables × 600 samples
- `samples/Data_var800_samples2000.npy`: 800 variables × 2000 samples
- `test_1000x2000.npy`: 1000 variables × 2000 samples

## 빌드 방법

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 벤치마크 실행

### 1. CPU 버전
```bash
./build/src-cpp/xges --input <data.npy>
```

### 2. GPU Custom Kernel
```bash
./build/src-cpp/xges --batch --input <data.npy>
```

### 3. GPU cuBLAS Library
```bash
./build/src-cpp/xges --batch --backend cublas --input <data.npy>
```

## 데이터 생성

```bash
# 1000 variables × 2000 samples
python generate_data_and_ground_truth.py --n-variables 1000 --n-samples 2000
```

## 벤치마크 결과 (실험 기준)

| Dataset | Vars × Samples | CPU | GPU Custom | GPU cuBLAS | Overhead |
|---------|----------------|-----|------------|------------|----------|
| Small   | 200 × 600      | 5.61s | 3.83s | 4.19s | +9.5% |
| Medium  | 800 × 2000     | 103.57s | 51.90s | 53.96s | +4.0% |
| Large   | 1000 × 2000    | 133.79s | 102.43s | 103.67s | +1.2% |

## 시스템 요구사항

- CUDA Toolkit 11.0+
- cuBLAS library
- CMake 3.18+
- Python 3.8+ (데이터 생성용)
- NumPy, NetworkX (데이터 생성용)

