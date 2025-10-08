### This project contains benchmarking results of the libraries that can parse YAML **and** also support Serde.

| Crate | Version | Merge Keys | Nested Enums | Duplicate key rejection | Notes |
|------:|:---------|:-----------|:--------------|:------------------------|:-------|
| [serde-saphyr](https://crates.io/crates/serde-saphyr) | 0.0.4 | ✅ Native | ✅ | ✅ Configurable          | No `unsafe`, no [unsafe-libyaml](https://crates.io/crates/unsafe-libyaml) |
| [serde-yaml-bw](https://crates.io/crates/serde-yaml_bw) | 2.4.1 | ✅ Native | ✅ | ✅ Configurable          | Slow due Saphyr doing budget check first upfront of libyaml |
| [serde-yaml-ng](https://crates.io/crates/serde-yaml-ng) | 0.10.0 | ⚠️ apply_merge | ❌ | ❌                       |  |
| [serde-yaml](https://crates.io/crates/serde-yaml) | 0.9.34 + deprecated | ⚠️ apply_merge | ❌ | ❌                       | Original, deprecated, repo archived |
| [serde-norway](https://crates.io/crates/serde-norway) | 0.9 | ⚠️ apply_merge | ❌ | ❌                       |  |
| [serde-yml](https://crates.io/crates/serde-yml) | 0.0.12 | ⚠️ apply_merge | ❌ | ❌                       | Repo archived |


**Note**: "apply_merge" indicates that the crate does not resolve YAML merge keys (`<<:`) automatically during deserialization right into Rust structure. Instead, the parsed `serde_yaml::Value` (or equivalent) must be passed through `.apply_merge()` to merge inherited mappings before only then converting to a typed struct. So partial support.

### System Configuration (Benchmark Environment)
Benchmarking of five libraries was done using code that can be found in benchmark/benches of this project, on the following hardware:

| **Category** | **Details**                                                                                                                                                  |
|:--------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **OS** | Ubuntu 22.04.5 LTS (`jammy`), Linux kernel 5.15.0-156-generic                                                                                                |
| **CPU** | Intel® Core™ i7-8700K @ 3.70 GHz (6 cores / 12 threads)                                                                                                      |
| **CPU Features** | AVX2, FMA, SSE4.2, AES, BMI1/2, HWP                                                                                                                          |
| **CPU Frequency Range** | 800 MHz – 5000 MHz                                                                                                                                           |
| **CPU Governor** | powersave                                                                                                                                                    |
| **Memory** | 64 GB DDR4 (2 × 32 GB @ 3000 MT/s)                                                                                                                           |
| **Swap** | Disabled (0 B)                                                                                                                                               |
| **Rust Toolchain** | rustc 1.90.0 (1159e78c4 2025-09-14)  •  cargo 1.90.0 (840b83a10 2025-07-30)                                                                                  |
| **Main Dependencies** | `serde 1.0.228`, `serde-saphyr 0.0.4`, `serde_norway 0.9.42`, `serde-yaml-ng 0.10.0`, `saphyr-parser 0.0.6`, `serde_json 1.0.145`, `smallvec 2.0.0-alpha.11` |

### Results
<p align="center">
<img src="https://github.com/bourumir-wyngs/serde-saphyr-benchmark/blob/master/figures/yaml_parse/relative_vs_baseline.png?raw=true"
alt="Relative median time vs baseline"
width="60%">
</p>

<p align="center">
  <img src="https://github.com/bourumir-wyngs/serde-saphyr-benchmark/blob/master/figures/yaml_parse/central_vs_size.png?raw=true" 
       alt="Central vs size" 
       width="60%">
</p>

<p align="center">
  <img src="https://github.com/bourumir-wyngs/serde-saphyr-benchmark/blob/master/figures/yaml_parse/throughput_vs_size.png?raw=true" 
       alt="Throughput vs size" 
       width="60%">
</p>

Data were processed with the script [plot_criterion.py](./plot_criterion.py) located in the root folder of this project. 
