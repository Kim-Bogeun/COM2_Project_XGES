# 실행 가이드

## 빌드

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 실행 방법 예시(600x200 데이터셋 예시, 파일 첨부)

### CPU 버전 (기본)
```bash
./build/src-cpp/xges --input samples/Data_var200_samples600.npy
```

### GPU 최적화 버전 (Custom Kernel)
```bash
./build/src-cpp/xges --batch --input samples/Data_var200_samples600.npy
```

### GPU cuBLAS 버전
```bash
./build/src-cpp/xges --batch --backend cublas --input samples/Data_var200_samples600.npy
```

## 성능 비교

두 버전 비교 코드:

```bash
# CPU 버전 실행 및 시간 측정
time ./build/src-cpp/xges --input samples/Data_var200_samples600.npy --output cpu_result.txt --stats cpu_stats.txt

# GPU 최적화 버전 실행 및 시간 측정
time ./build/src-cpp/xges --batch --input samples/Data_var200_samples600.npy --output gpu_result.txt --stats gpu_stats.txt
```

`cpu_stats.txt`, `gpu_stats.txt`에서 실행 시간과 성능 확인 가능
