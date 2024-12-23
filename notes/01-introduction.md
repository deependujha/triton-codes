# An introduction to Triton

- While programming in triton, think in terms of block and not individual thread.

---

## Load & Store tensor in triton

1. `triton.language.load`: Return a tensor of data whose values are loaded from memory at location defined by pointer.

2. `triton.language.store`: Store a tensor of data into memory locations defined by pointer.

> **Mask**: store & load accept an argument `mask`, which if true for an index, does the operation (store/load), else `false` ignores for that index. Don't store/load if mask[idx]=False.

---

## Triton JIT

- to mark a function to be executed by cuda (accelerated device), decorate it with `@triton.jit`.

---

## `tl.arange(start, end)`

- return value from [start, end). Both, start and end must be power of 2, and end>start.

---

## Triton constexpr

- `tl.constexpr` is a special annotation used to indicate that a function parameter must be a **`compile-time constant`**.

```python
import triton
import triton.language as tl

@triton.jit
def my_fn(BLOCK_SIZE: tl.constexpr)
    ...
```

---

## Triton `program_id` & `num_programs`

- triton launches programs in 3D.

- `tl.program_id(axis)`: Gets the program’s ID along axis.
- `tl.num_programs(axis)`: Gets the total programs along axis.

- `axis can only be (0,1,2)`

```python
pid = tl.program_id(0)  # Current program ID on axis 0
num_progs = tl.num_programs(0)  # Total programs on axis 0
```

---

## Launching a cuda kernel (`calling triton jitted function`)

- To laucn a kernel, we need to specify the dimensions on which we are launching.
- `meta keyword in lambda fn` is a dictionary passed by triton, that contains all the parameter which we pass while calling the kernel function.
- Prefer to mark those arguments as `constexpr` in triton kernel function.

```python
@triton.jit
def some_kernel_fn(param1, param2,SOME_KEYWORD: tl.constexpr, ...):
    ...

def calling_kernel():
    grid = lambda meta: (meta['SOME_KEYWORD']), ) # meta contains all the values that we passed while calling the kernel code. Prefer to mark those arguments in kernel as `constexpr`.
    some_kernel_fn[grid](arg1, arg2, SOME_KEYWORD=4, ...)

```

- kernel program gets launched on this grid.
- To identify, which block are we on, and the segment of the vector that we have to work in is given by: `tl.program_id(axis)`.

---

## Simple Vector Addition in triton

```python
import torch
import triton
import triton.language as tl

DEVICE = "cuda"

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)

    assert x.device == y.device == output.device and x.device.type == "cuda"

    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

def main():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')

if __name__ == "__main__":
    main()
```
