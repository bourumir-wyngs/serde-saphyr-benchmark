### System Configuration (Benchmark Environment)
Benchmarking of five libraries was done using code that can be found in benchmark/benches of this project, on the following hardware:

| **Category** | **Details** |
|:--------------|:------------|
| **OS** | Ubuntu 22.04.5 LTS (`jammy`), Linux kernel 5.15.0-156-generic |
| **CPU** | Intel® Core™ i7-8700K @ 3.70 GHz (6 cores / 12 threads) |
| **CPU Features** | AVX2, FMA, SSE4.2, AES, BMI1/2, HWP |
| **CPU Frequency Range** | 800 MHz – 5000 MHz |
| **CPU Governor** | powersave |
| **Memory** | 64 GB DDR4 (2 × 32 GB @ 3000 MT/s) |
| **Swap** | Disabled (0 B) |
| **Rust Toolchain** | rustc 1.90.0 (1159e78c4 2025-09-14)  •  cargo 1.90.0 (840b83a10 2025-07-30) |
| **Main Dependencies** | `serde 1.0.228`, `serde-saphyr 0.0.4`, `serde_norway 0.9.42`, `saphyr-parser 0.0.6`, `serde_json 1.0.145`, `smallvec 2.0.0-alpha.11` |


<p align="center">
<img src="https://github.com/bourumir-wyngs/serde-saphyr-benchmark/blob/master/figures/relative_vs_baseline.png?raw=true"
alt="Relative median time vs baseline"
width="40%">
</p>

<p align="center">
  <img src="https://github.com/bourumir-wyngs/serde-saphyr-benchmark/blob/master/figures/central_vs_size.png?raw=true" 
       alt="Central vs size" 
       width="40%">
</p>

<p align="center">
  <img src="https://github.com/bourumir-wyngs/serde-saphyr-benchmark/blob/master/figures/throughput_vs_size.png?raw=true" 
       alt="Throughput vs size" 
       width="40%">
</p>

Data were processed with the script [plot_criterion.py](./plot_criterion.py) located in the root folder of this project. Raw benchmark data as produced by Criterion are available in the archive [criterion.zip](./criterion.zip).
