We support the LAMMPS pair style `d3` of the Grimme's D3 dispersion (van der Waals) correction scheme accelerated with CUDA, which can be used within LAMMPS in conjunction with SevenNet.

**PLEASE NOTE:** `d3` only runs on a single MPI rank / single GPU. For multiple MPI ranks or GPUs, use `d3/mpi`
(or `d3sycl/mpi` for the SYCL build); see [Multi-GPU / multi-rank](#multi-gpu--multi-rank-d3mpi-and-d3syclmpi).

# About Grimme's D3 code accelerated with CUDA

This is LAMMPS implementation of [Grimme's D3 method](https://doi.org/10.1063/1.3382344). We have ported the code from the [original fortran code](https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3) to a LAMMPS pair style written in CUDA/C++.

While D3 method is significantly faster than DFT, existing CPU implementations were slower than SevenNet. To address this, we have adopted CUDA and single precision (FP32) operations to accelerate the code.

## Installation for LAMMPS

Simply run,
```bash
sevenn_patch_lammps ./lammps_sevenn --d3
```

You can follow the remaining installation steps in the [SevenNet documentation](../../README.md#installation-for-lammps).

Also, this code requires a GPU with a compute capability of **at least 6.0**. If you try to compile it with version 5.0, you may encounter an `atomicAdd` error.

The target compute capability of this code follows the setting of LibTorch in SevenNet, except for version 5.0.

You can manually select the target capability using the `TORCH_CUDA_ARCH_LIST` environment variable. For example, you can use: `export TORCH_CUDA_ARCH_LIST="6.1;7.0;8.0;8.6;8.9;9.0"`.

The `--d3` patch also installs the multi-GPU style `d3/mpi` (`pair_d3_mpi.h`/`pair_d3_mpi.cu`) in the same binary.

### SYCL build (`d3sycl`, `d3sycl/mpi`)

A SYCL port of this pair style, `d3sycl`, is also provided (`pair_d3sycl.h`/`pair_d3sycl.cpp`),
together with the SYCL port of `d3/mpi`, `d3sycl/mpi` (`pair_d3sycl_mpi.h`/`pair_d3sycl_mpi.cpp`),
built with [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) instead of raw CUDA. It
implements the same math and accepts the same `pair_style`/`pair_coeff` arguments as `d3`
(just substitute `d3sycl` for `d3`), so it can be used as a drop-in alternative on systems
where a SYCL toolchain is preferred over (or more readily available than) CUDA/nvcc, while
still running on NVIDIA GPUs via AdaptiveCpp's CUDA backend.

To patch it in, run
```bash
sevenn_patch_lammps ./lammps_sevenn --d3sycl
```
or call the script directly with `1` as the 6th argument:
```bash
sh patch_lammps.sh ./lammps_sevenn 17 0 NONE NONE 1
```
and configure LAMMPS with a SYCL compiler. With AdaptiveCpp, make it discoverable via `CMAKE_PREFIX_PATH`
(alongside LibTorch):
```bash
cmake ../cmake \
    -DCMAKE_PREFIX_PATH="/path/to/adaptivecpp/install;$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
    -DACPP_TARGETS=generic
```
With AdaptiveCpp, `BUILD_OMP` is forced off: AdaptiveCpp compiles all LAMMPS sources with its clang, whose
OpenMP runtime (`libomp`) clashes with the host compiler's `libgomp` at link time.

With Intel oneAPI DPC++, source the oneAPI environment and select `icpx` as the C++ compiler:
```bash
source /opt/intel/oneapi/setvars.sh
cmake ../cmake -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
```

**`d3sycl` (SYCL) and `d3` (CUDA) cannot be patched into the same LAMMPS source tree** -- they
use incompatible compilation flows (`add_sycl_to_target()` wraps the whole `lammps` target's
compile rules, including the `.cu` files compiled by `nvcc` for `d3`). Patch each into its own
LAMMPS checkout if you need both. `patch_lammps.sh` will refuse to proceed if both are
requested at once.

Internally, `d3sycl` flattens the nested device arrays used by the CUDA implementation into
flat USM (`sycl::malloc_shared`) buffers, and uses `sycl::atomic_ref` adds in place of the CUDA
version's block-local shared-memory reduction for the virial/dispersion-energy accumulation --
simpler and numerically equivalent, at a small performance cost.

## Usage for LAMMPS

You can use the D3 dispersion correction in LAMMPS with SevenNet through the `pair/hybrid` command:

```txt
pair_style hybrid/overlay e3gnn d3 {sq_r_cut} {sq_r_cn_cut} {type_of_damping} {name_of_functional}
pair_coeff * * e3gnn {path_to_serial_model} {space_separated_chemical_species}
pair_coeff * * d3 {space_separated_chemical_species}
```

for example,

```txt
pair_style hybrid/overlay e3gnn d3 9000 1600 damp_bj pbe
pair_coeff * * e3gnn ./deployed_serial.pt C H O
pair_coeff * * d3 C H O
```

`sq_r_cut` and `sq_r_cn_cut` are **square of cutoff radii** for energy/force calculation and coordination number, respectively. Units are squared Bohr, where 1 Bohr = 0.52917721 Å.
Default values are `9000` and `1600` (Bohr²), which correspond to `50.2022` `21.1671`, respectively. This is also the default values used in VASP.[^1]

Available `type_of_damping` are as follows:
- `damp_zero`: Zero damping
- `damp_bj`: Becke-Johnson damping

Available `name_of_functional` options are the same as in the original Fortran code. SevenNet-0 is trained on the 'PBE' functional, so you should specify 'pbe' in the script when using it. For other supporting functionals, check 'List of parametrized functionals' in [here](https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3). We are actively updating the D3 parameters for other functionals, such as `r2scan` and `wb97m`.

To use the SYCL build instead, substitute `d3sycl` for `d3` in both the `pair_style` and `pair_coeff` lines above; all arguments stay the same.

## Multi-GPU / multi-rank (`d3/mpi` and `d3sycl/mpi`)

`d3` and `d3sycl` are only correct on a single MPI rank. `d3/mpi` (CUDA) and `d3sycl/mpi` (SYCL) take the same
arguments and can run on any number of MPI ranks, each rank using its own GPU (node-local rank modulo the number of
visible GPUs):
```
pair_style hybrid/overlay e3gnn/parallel d3/mpi {sq_r_cut} {sq_r_cn_cut} {type_of_damping} {name_of_functional}
pair_coeff * * d3/mpi {element_C} {element_H} {element_O}
```
Every rank gathers all atom positions and evaluates its slice of the N(N+1)/2 pair list on its GPU (64-bit pair
indices, so N > 65,536 works). Coordination numbers and dE/dCN are combined with `MPI_Allreduce`, forces with
`MPI_Reduce_scatter`. Compared with `d3`, the MPI styles also:
- allow several atom types to map to the same element (`d3` gives inf/nan),
- do not wrap atoms along non-periodic dimensions,
- accumulate the total energy in double precision,
- raise an error for `damp_zerom`/`damp_bjm` (only `damp_zero` and `damp_bj` are supported).

Set `D3_MPI_TIMING=1` (`mpirun -x D3_MPI_TIMING=1 ...`) to print a per-rank timing breakdown at exit. For
`d3sycl/mpi` on NVIDIA GPUs with AdaptiveCpp, export `ACPP_VISIBILITY_MASK=cuda` and forward it to all ranks
(`mpirun -x ACPP_VISIBILITY_MASK ...`); otherwise it may fall back to the OpenMP CPU device.

## Features
- Selective(or no) periodic boundary condition: implemented, But only PBC/noPBC can be checked through original FORTRAN code; selective PBC cannot
- 3-body term, n > 8 term: not implemented (as to VASP)
- Modified versions of zero and bj damping

## Cautions
- It can be slower than the CPU with a small number of atoms.
- The maximum number of atoms that can be calculated is 46,340 (overflow issue). `d3/mpi` and `d3sycl/mpi` use 64-bit pair indices and do not have this limit.
- There can be occurred small amounts of numerical error
  - The introduction of some FP32 operations can lead to minor numerical errors, particularly in pressure calculations, but these are generally smaller than those seen with SevenNet.
  - If the error is too large, ensure that the `fmad=false` option in `patch_lammps.sh` is correctly applied during build.

## To do
- Remove atom_modify / compute virial dependency.
- Add support for ASE as calculator interface.
- Add support for multi GPUs (with `e3gnn/parallel`). Done via `d3/mpi` / `d3sycl/mpi`.
- Implement without Unified Memory.
- Unfix the `threadsPerBlock=128`.
- Unroll the repetition loop `k` (for small number of atoms).

## Contributors
- Hyungmin An: Ported the original Fortran D3 code to C++ with OpenMP and MPI.
- Gijin Kim: Accelerated the C++ D3 code with OpenACC[^2] and CUDA, and currently maintains it.
- @alvarovm:  `d3sycl`, SYCL (tested with AdaptiveCpp) port of the CUDA `d3` pair style; multi-GPU `d3/mpi` and its SYCL port `d3sycl/mpi`.

[^1]: On the [VASP DFT-D3](https://www.vasp.at/wiki/index.php/DFT-D3) page, the `VDW_RADIUS` and `VDW_CNRADIUS` are `50.2` and `20.0`, respectively (units are Å). However, when running VASP 6.3.2, the default values in the OUTCAR file are `50.2022` and `21.1671`. These values are the same as our defaults.
[^2]: Since OpenACC is not compatible with libtorch, we chose to use the CUDA.
