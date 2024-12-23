# Triton Codes

- Triton is a language and compiler for parallel programming.
- It aims to provide a Python-based programming environment for productively writing custom DNN compute kernels capable of running at maximal throughput on modern GPU hardware.

---

## Installation

```bash
pip install triton
```

---

## What makes it different from normal cuda programming?


| **Aspect**            | **CUDA Programming**                             | **Triton Programming**                          |
|------------------------|-------------------------------------------------|------------------------------------------------|
| **Focus**             | Thread-centric                                   | Block-centric                                  |
| **Execution Logic**   | Write logic for an individual thread             | Write logic for a block of threads             |
| **Thread Management** | Explicit: Use `threadIdx`, `blockIdx`, etc.      | Implicit: Handled by Triton                    |
| **Data Indexing**     | Calculate global indices manually                | Operate on tensors representing data blocks    |
| **Abstraction Level** | Low-level: Direct control over hardware threads  | High-level: Abstracts hardware thread details  |
| **Optimization**      | Fine-grained control over memory and threads     | High-level optimizations applied automatically |
| **Ease of Use**       | Requires more manual effort for setup and tuning | Easier and faster development                  |
| **Best Use Case**     | Highly customized, thread-specific optimizations | Block-level parallelism like matrix operations |

---

## Triton to PTX (parallel thread execution)

- when we run triton, cuda code is not generated. But rather, PTX (parallel thread execution) code is generated which is lower level than cuda code.

---
