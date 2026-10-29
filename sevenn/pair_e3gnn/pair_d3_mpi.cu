/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Multi-GPU (MPI) variant of pair_style d3.
   Original CUDA D3 implementation: Gijin Kim, Hyungmin An (SNU)

   Scheme (replicated data, pair decomposition):
     1. MPI_Allgatherv of local positions/types/IDs -> every rank holds
        all N atoms, ordered by atom ID.
     2. Rank r evaluates pairs [r*M/P, (r+1)*M/P) of the M = N(N+1)/2
        lower-triangular pair list on its own GPU.
     3. CN and dE/dCN are combined with MPI_Allreduce (N doubles each),
        forces with MPI_Reduce_scatter (each rank receives its local atoms).
        Energy and virial stay per-rank partial sums (LAMMPS sums them).
------------------------------------------------------------------------- */

#include "pair_d3_mpi.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "utils.h"

#include "pair_d3_pars.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <unordered_map>

using namespace LAMMPS_NS;

#define D3_THREADS 128
#define D3_MAXC 5
static_assert(D3_MAXC == PairD3MPI::MAXC, "D3_MAXC mismatch");

#define CUDA_CHECK(call)                                                                       \
  do {                                                                                         \
    cudaError_t status_ = (call);                                                              \
    if (status_ != cudaSuccess)                                                                \
      error->one(FLERR, "pair d3/mpi CUDA error: {} ({})", cudaGetErrorString(status_), #call); \
  } while (0)

/* ----------------------------------------------------------------------
   Device helpers
------------------------------------------------------------------------- */

// k-th entry of the lower-triangular pair list (i >= j), 64-bit safe
__device__ __forceinline__ void pair_from_index(long long k, int &i, int &j)
{
  long long ii = static_cast<long long>((sqrt(8.0 * static_cast<double>(k) + 1.0) - 1.0) * 0.5);
  while (ii * (ii + 1) / 2 > k) --ii;
  while ((ii + 1) * (ii + 2) / 2 <= k) ++ii;
  i = static_cast<int>(ii);
  j = static_cast<int>(k - ii * (ii + 1) / 2);
}

__device__ __forceinline__ float lensq3f(const float *v)
{
  return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

// block-wide sum of NV per-thread values, one double atomicAdd per value and block
template <int NV> __device__ __forceinline__ void block_reduce_add(const float (&v)[NV], double *out)
{
  __shared__ float sh[NV][D3_THREADS];
  for (int q = 0; q < NV; q++) sh[q][threadIdx.x] = v[q];
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      for (int q = 0; q < NV; q++) sh[q][threadIdx.x] += sh[q][threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    for (int q = 0; q < NV; q++) atomicAdd(&out[q], static_cast<double>(sh[q][0]));
}

/* ----------------------------------------------------------------------
   Coordination numbers
------------------------------------------------------------------------- */

__global__ void kernel_cn(long long lo, long long hi, int nimg, int self_img, float cnthr, float K1,
                          const float *__restrict__ rcov, const float *__restrict__ tau,
                          const int *__restrict__ type, const float *__restrict__ x, double *cn)
{
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long p = lo + static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; p < hi;
       p += stride) {
    int iat, jat;
    pair_from_index(p, iat, jat);
    float cn_local = 0.0f;

    if (iat == jat) {
      const float rcov_sum = rcov[type[iat]] * 2.0f;
      for (int k = nimg - 1; k >= 0; k--) {
        if (k == self_img) continue;
        const float rx = tau[3 * k];
        const float ry = tau[3 * k + 1];
        const float rz = tau[3 * k + 2];
        const float r2 = rx * rx + ry * ry + rz * rz;
        if (r2 <= cnthr) {
          const float r_rc = rsqrtf(r2);
          cn_local += 1.0f / (1.0f + expf(-K1 * ((rcov_sum * r_rc) - 1.0f)));
        }
      }
      atomicAdd(&cn[iat], cn_local);
    } else {
      const float rcov_sum = rcov[type[iat]] + rcov[type[jat]];
      const float dx = x[3 * jat] - x[3 * iat];
      const float dy = x[3 * jat + 1] - x[3 * iat + 1];
      const float dz = x[3 * jat + 2] - x[3 * iat + 2];
      for (int k = nimg - 1; k >= 0; k--) {
        const float rx = dx + tau[3 * k];
        const float ry = dy + tau[3 * k + 1];
        const float rz = dz + tau[3 * k + 2];
        const float r2 = rx * rx + ry * ry + rz * rz;
        if (r2 <= cnthr) {
          const float r_rc = rsqrtf(r2);
          cn_local += 1.0f / (1.0f + expf(-K1 * ((rcov_sum * r_rc) - 1.0f)));
        }
      }
      atomicAdd(&cn[iat], cn_local);
      atomicAdd(&cn[jat], cn_local);
    }
  }
}

/* ----------------------------------------------------------------------
   C6(CN_i, CN_j) and its derivatives w.r.t. CN_i, CN_j for one pair
   (computed on the fly instead of being stored for all N^2/2 pairs)
------------------------------------------------------------------------- */

__device__ __forceinline__ void get_c6(int ti, int tj, float cni, float cnj, int np1, float K3,
                                       const int *__restrict__ mxc, const float *__restrict__ c6ab,
                                       float &c6, float &dc6iji, float &dc6ijj)
{
  const int mxci = mxc[ti];
  const int mxcj = mxc[tj];
  const float *base = c6ab + static_cast<size_t>(ti * np1 + tj) * D3_MAXC * D3_MAXC * 3;

  float c6mem = -INFINITY;
  float r_save = 9999.0f;
  double numerator = 0.0;
  double denominator = 0.0;
  double d_numerator_i = 0.0;
  double d_denominator_i = 0.0;
  double d_numerator_j = 0.0;
  double d_denominator_j = 0.0;

  for (int a = 0; a < mxci; a++) {
    for (int b = 0; b < mxcj; b++) {
      const float *ref = base + (a * D3_MAXC + b) * 3;
      const float c6ref = ref[0];
      if (c6ref > 0.0f) {
        const float cn_refi = ref[1];
        const float cn_refj = ref[2];
        const float r = (cn_refi - cni) * (cn_refi - cni) + (cn_refj - cnj) * (cn_refj - cnj);
        if (r < r_save) {
          r_save = r;
          c6mem = c6ref;
        }
        double expterm = exp(static_cast<double>(K3) * static_cast<double>(r));
        numerator += c6ref * expterm;
        denominator += expterm;

        expterm *= 2.0f * K3;

        double term = expterm * (cni - cn_refi);
        d_numerator_i += c6ref * term;
        d_denominator_i += term;

        term = expterm * (cnj - cn_refj);
        d_numerator_j += c6ref * term;
        d_denominator_j += term;
      }
    }
  }

  if (denominator > 1e-99) {
    const double denominator_rc = 1.0 / denominator;
    const double unit_frac = numerator * denominator_rc;
    c6 = unit_frac;
    dc6iji = denominator_rc * fma(unit_frac, -d_denominator_i, d_numerator_i);
    dc6ijj = denominator_rc * fma(unit_frac, -d_denominator_j, d_numerator_j);
  } else {
    c6 = c6mem;
    dc6iji = 0.0f;
    dc6ijj = 0.0f;
  }
}

/* ----------------------------------------------------------------------
   Two-body dispersion term for one image.
   BJ = 0: zero damping, pc = {a1*r0, a2*r0, s8*r42}
   BJ = 1: BJ damping,   pc = {R0^6, R0^8, s8*r42*3}
   SELF:   interaction of an atom with its own periodic images (factor 1/2)
------------------------------------------------------------------------- */

template <int BJ, bool SELF>
__device__ __forceinline__ void d3_term(const float *rij, float r2, float c6, float pc1, float pc2,
                                        float s8r, float s6, float alp6, float alp8, float *vec,
                                        float &dc6_rest)
{
  if (BJ == 0) {
    const float r_rc = rsqrtf(r2);
    const float unit_rc_a1 = pc1 * r_rc;
    float t6 = unit_rc_a1 * unit_rc_a1;    // ^2
    t6 *= unit_rc_a1;                      // ^3
    t6 *= t6;                              // ^6
    t6 *= unit_rc_a1;                      // ^7
    t6 *= t6;                              // ^14
    const float damp6 = 1.0f / fmaf(t6, 6.0f, 1.0f);
    const float unit_rc_a2 = pc2 * r_rc;
    float t8 = unit_rc_a2 * unit_rc_a2;    // ^2
    t8 *= t8;                              // ^4
    t8 *= t8;                              // ^8
    t8 *= t8;                              // ^16
    const float damp8 = 1.0f / fmaf(t8, 6.0f, 1.0f);
    const float r2_rc = r_rc * r_rc;
    const float r6_rc = r2_rc * r2_rc * r2_rc;
    const float r8_rc = r6_rc * r2_rc;
    const float x1 = (SELF ? 3.0f : 6.0f) * c6 * r8_rc *
        fmaf(r2_rc, s8r * damp8 * fmaf(3.0f * alp8 * t8, damp8, -4.0f),
             s6 * damp6 * fmaf(alp6 * t6, damp6, -1.0f));
    vec[0] = x1 * rij[0];
    vec[1] = x1 * rij[1];
    vec[2] = x1 * rij[2];
    if (SELF)
      dc6_rest = 0.5f * r6_rc * fmaf(3.0f * r2_rc, s8r * damp8, s6 * damp6);
    else
      dc6_rest = r6_rc * fmaf(3.0f * r2_rc, s8r * damp8, s6 * damp6);
  } else {
    const float r = sqrtf(r2);
    const float r5 = r2 * r2 * r;
    const float r7 = r5 * r2;
    const float t6_rc = 1.0f / fmaf(r5, r, pc1);
    const float t8_rc = 1.0f / fmaf(r7, r, pc2);
    const float t6_sqrc = t6_rc * t6_rc;
    const float t8_sqrc = t8_rc * t8_rc;
    float x1;
    if (SELF)
      x1 = -c6 * fmaf(4.0f * s8r * r7, t8_sqrc, 3.0f * s6 * r5 * t6_sqrc);
    else
      x1 = -c6 * fmaf(8.0f * s8r * r7, t8_sqrc, 6.0f * s6 * r5 * t6_sqrc);
    const float r_rc = 1.0f / r;
    vec[0] = x1 * rij[0] * r_rc;
    vec[1] = x1 * rij[1] * r_rc;
    vec[2] = x1 * rij[2] * r_rc;
    if (SELF)
      dc6_rest = 0.5f * fmaf(s8r, t8_rc, s6 * t6_rc);
    else
      dc6_rest = fmaf(s8r, t8_rc, s6 * t6_rc);
  }
}

/* ----------------------------------------------------------------------
   Dispersion energy, forces at fixed C6, virial and dE/dCN (dc6i)
   red[0..8] = sigma, red[9] = energy
------------------------------------------------------------------------- */

template <int BJ>
__global__ void kernel_vdw(long long lo, long long hi, int nimg, int self_img, int np1, float rthr,
                           float s6, float s8, float a1, float a2, float alp6, float alp8, float K3,
                           const float *__restrict__ r2r4, const float *__restrict__ r0ab,
                           const int *__restrict__ mxc, const float *__restrict__ c6ab,
                           const float *__restrict__ tau, const int *__restrict__ type,
                           const float *__restrict__ x, const double *__restrict__ cn, double *dc6i,
                           double *f, double *red)
{
  float acc[10] = {0.0f};    // sigma (9) + energy
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;

  for (long long p = lo + static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; p < hi;
       p += stride) {
    int iat, jat;
    pair_from_index(p, iat, jat);
    const int ti = type[iat];
    const int tj = type[jat];

    float c6, dc6iji, dc6ijj;
    get_c6(ti, tj, static_cast<float>(cn[iat]), static_cast<float>(cn[jat]), np1, K3, mxc, c6ab,
           c6, dc6iji, dc6ijj);

    float pc1, pc2, s8r;
    if (BJ == 0) {
      const float r0 = r0ab[ti * np1 + tj];
      pc1 = a1 * r0;
      pc2 = a2 * r0;
      s8r = s8 * (r2r4[ti] * r2r4[tj]);
    } else {
      const float r42x3 = r2r4[ti] * r2r4[tj] * 3.0f;
      const float R0 = fmaf(a1, sqrtf(r42x3), a2);
      const float R0_2 = R0 * R0;
      pc1 = R0_2 * R0_2 * R0_2;
      pc2 = pc1 * R0_2;
      s8r = s8 * r42x3;
    }

    float f_local[3] = {0.0f, 0.0f, 0.0f};
    float dc6i_local_i = 0.0f;
    float dc6i_local_j = 0.0f;
    float vec[3], dc6_rest;

    if (iat == jat) {
      for (int k = nimg - 1; k >= 0; k--) {
        if (k == self_img) continue;
        const float rij[3] = {tau[3 * k], tau[3 * k + 1], tau[3 * k + 2]};
        const float r2 = lensq3f(rij);
        if (r2 > rthr) continue;
        d3_term<BJ, true>(rij, r2, c6, pc1, pc2, s8r, s6, alp6, alp8, vec, dc6_rest);
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) acc[3 * a + b] += vec[a] * rij[b];
        acc[9] -= dc6_rest * c6;
        dc6i_local_i += dc6_rest * dc6iji;
        dc6i_local_j += dc6_rest * dc6ijj;
      }
      atomicAdd(&dc6i[iat], dc6i_local_i);
      atomicAdd(&dc6i[jat], dc6i_local_j);
    } else {
      const float dx = x[3 * jat] - x[3 * iat];
      const float dy = x[3 * jat + 1] - x[3 * iat + 1];
      const float dz = x[3 * jat + 2] - x[3 * iat + 2];
      for (int k = nimg - 1; k >= 0; k--) {
        const float rij[3] = {dx + tau[3 * k], dy + tau[3 * k + 1], dz + tau[3 * k + 2]};
        const float r2 = lensq3f(rij);
        if (r2 > rthr) continue;
        d3_term<BJ, false>(rij, r2, c6, pc1, pc2, s8r, s6, alp6, alp8, vec, dc6_rest);
        f_local[0] -= vec[0];
        f_local[1] -= vec[1];
        f_local[2] -= vec[2];
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) acc[3 * a + b] += vec[a] * rij[b];
        acc[9] -= dc6_rest * c6;
        dc6i_local_i += dc6_rest * dc6iji;
        dc6i_local_j += dc6_rest * dc6ijj;
      }
      atomicAdd(&dc6i[iat], dc6i_local_i);
      atomicAdd(&dc6i[jat], dc6i_local_j);
      atomicAdd(&f[3 * iat], f_local[0]);
      atomicAdd(&f[3 * iat + 1], f_local[1]);
      atomicAdd(&f[3 * iat + 2], f_local[2]);
      atomicAdd(&f[3 * jat], -f_local[0]);
      atomicAdd(&f[3 * jat + 1], -f_local[1]);
      atomicAdd(&f[3 * jat + 2], -f_local[2]);
    }
  }

  block_reduce_add<10>(acc, red);
}

/* ----------------------------------------------------------------------
   Forces and virial from the CN dependence of C6 (needs global dc6i)
------------------------------------------------------------------------- */

__global__ void kernel_dc6(long long lo, long long hi, int nimg, int self_img, float cnthr, float K1,
                           const double *__restrict__ dc6i, const float *__restrict__ rcov,
                           const float *__restrict__ tau, const int *__restrict__ type,
                           const float *__restrict__ x, double *f, double *red)
{
  float acc[9] = {0.0f};
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;

  for (long long p = lo + static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; p < hi;
       p += stride) {
    int iat, jat;
    pair_from_index(p, iat, jat);

    if (iat == jat) {
      const float rcov_sum = rcov[type[iat]] * 2.0f;
      const float dc6i_sum = dc6i[iat];
      for (int k = nimg - 1; k >= 0; k--) {
        if (k == self_img) continue;
        const float rij[3] = {tau[3 * k], tau[3 * k + 1], tau[3 * k + 2]};
        const float r2 = lensq3f(rij);
        if (r2 >= cnthr) continue;
        const float r_rc = rsqrtf(r2);
        const float expterm = expf(-K1 * (rcov_sum * r_rc - 1.0f));
        const float unit_rc = 1.0f / (r2 * (expterm + 1.0f) * (expterm + 1.0f));
        const float dcnn = -K1 * rcov_sum * expterm * unit_rc;
        const float x1 = dcnn * dc6i_sum;
        const float vec[3] = {x1 * rij[0] * r_rc, x1 * rij[1] * r_rc, x1 * rij[2] * r_rc};
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) acc[3 * a + b] += vec[a] * rij[b];
      }
    } else {
      const float rcov_sum = rcov[type[iat]] + rcov[type[jat]];
      const float dc6i_sum = dc6i[iat] + dc6i[jat];
      const float dx = x[3 * jat] - x[3 * iat];
      const float dy = x[3 * jat + 1] - x[3 * iat + 1];
      const float dz = x[3 * jat + 2] - x[3 * iat + 2];
      float f_local[3] = {0.0f, 0.0f, 0.0f};
      for (int k = nimg - 1; k >= 0; k--) {
        const float rij[3] = {dx + tau[3 * k], dy + tau[3 * k + 1], dz + tau[3 * k + 2]};
        const float r2 = lensq3f(rij);
        if (r2 >= cnthr) continue;
        const float r_rc = rsqrtf(r2);
        const float expterm = expf(-K1 * (rcov_sum * r_rc - 1.0f));
        const float unit_rc = 1.0f / (r2 * (expterm + 1.0f) * (expterm + 1.0f));
        const float dcnn = -K1 * rcov_sum * expterm * unit_rc;
        const float x1 = dcnn * dc6i_sum;
        const float vec[3] = {x1 * rij[0] * r_rc, x1 * rij[1] * r_rc, x1 * rij[2] * r_rc};
        f_local[0] -= vec[0];
        f_local[1] -= vec[1];
        f_local[2] -= vec[2];
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) acc[3 * a + b] += vec[a] * rij[b];
      }
      atomicAdd(&f[3 * iat], f_local[0]);
      atomicAdd(&f[3 * iat + 1], f_local[1]);
      atomicAdd(&f[3 * iat + 2], f_local[2]);
      atomicAdd(&f[3 * jat], -f_local[0]);
      atomicAdd(&f[3 * jat + 1], -f_local[1]);
      atomicAdd(&f[3 * jat + 2], -f_local[2]);
    }
  }

  block_reduce_add<9>(acc, red);
}

static int launch_blocks(long long npairs)
{
  const long long nb = (npairs + D3_THREADS - 1) / D3_THREADS;
  return static_cast<int>(std::min<long long>(std::max<long long>(nb, 1), 2147483647LL));
}

/* ----------------------------------------------------------------------
   Reference tables
------------------------------------------------------------------------- */

static const double r2r4_ref[94] = {
         2.00734898,  1.56637132,  5.01986934,  3.85379032,  3.64446594,
         3.10492822,  2.71175247,  2.59361680,  2.38825250,  2.21522516,
         6.58585536,  5.46295967,  5.65216669,  4.88284902,  4.29727576,
         4.04108902,  3.72932356,  3.44677275,  7.97762753,  7.07623947,
         6.60844053,  6.28791364,  6.07728703,  5.54643096,  5.80491167,
         5.58415602,  5.41374528,  5.28497229,  5.22592821,  5.09817141,
         6.12149689,  5.54083734,  5.06696878,  4.87005108,  4.59089647,
         4.31176304,  9.55461698,  8.67396077,  7.97210197,  7.43439917,
         6.58711862,  6.19536215,  6.01517290,  5.81623410,  5.65710424,
         5.52640661,  5.44263305,  5.58285373,  7.02081898,  6.46815523,
         5.98089120,  5.81686657,  5.53321815,  5.25477007, 11.02204549,
        10.15679528,  9.35167836,  9.06926079,  8.97241155,  8.90092807,
         8.85984840,  8.81736827,  8.79317710,  7.89969626,  8.80588454,
         8.42439218,  8.54289262,  8.47583370,  8.45090888,  8.47339339,
         7.83525634,  8.20702843,  7.70559063,  7.32755997,  7.03887381,
         6.68978720,  6.05450052,  5.88752022,  5.70661499,  5.78450695,
         7.79780729,  7.26443867,  6.78151984,  6.67883169,  6.39024318,
         6.09527958, 11.79156076, 11.10997644,  9.51377795,  8.67197068,
         8.77140725,  8.65402716,  8.53923501,  8.85024712
    }; // atomic <r^2>/<r^4> values

static const double rcov_ref[94] = {
        0.80628308, 1.15903197, 3.02356173, 2.36845659, 1.94011865,
        1.88972601, 1.78894056, 1.58736983, 1.61256616, 1.68815527,
        3.52748848, 3.14954334, 2.84718717, 2.62041997, 2.77159820,
        2.57002732, 2.49443835, 2.41884923, 4.43455700, 3.88023730,
        3.35111422, 3.07395437, 3.04875805, 2.77159820, 2.69600923,
        2.62041997, 2.51963467, 2.49443835, 2.54483100, 2.74640188,
        2.82199085, 2.74640188, 2.89757982, 2.77159820, 2.87238349,
        2.94797246, 4.76210950, 4.20778980, 3.70386304, 3.50229216,
        3.32591790, 3.12434702, 2.89757982, 2.84718717, 2.84718717,
        2.72120556, 2.89757982, 3.09915070, 3.22513231, 3.17473967,
        3.17473967, 3.09915070, 3.32591790, 3.30072128, 5.26603625,
        4.43455700, 4.08180818, 3.70386304, 3.98102289, 3.95582657,
        3.93062995, 3.90543362, 3.80464833, 3.82984466, 3.80464833,
        3.77945201, 3.75425569, 3.75425569, 3.72905937, 3.85504098,
        3.67866672, 3.45189952, 3.30072128, 3.09915070, 2.97316878,
        2.92277614, 2.79679452, 2.82199085, 2.84718717, 3.32591790,
        3.27552496, 3.27552496, 3.42670319, 3.30072128, 3.47709584,
        3.57788113, 5.06446567, 4.56053862, 4.20778980, 3.98102289,
        3.82984466, 3.85504098, 3.88023730, 3.90543362
    }; // covalent radii

/* ---------------------------------------------------------------------- */

PairD3MPI::PairD3MPI(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;

  damping = 1;
  rthr = cnthr = 0.0f;
  s6 = s8 = s18 = rs6 = rs8 = rs18 = alp = alp6 = alp8 = a1 = a2 = 0.0f;

  d_r2r4 = d_rcov = d_r0ab = d_c6ab = nullptr;
  d_mxc = nullptr;

  nimg_vdw = nimg_cn = 0;
  rep_vdw[0] = rep_vdw[1] = rep_vdw[2] = 0;
  rep_cn[0] = rep_cn[1] = rep_cn[2] = 0;
  d_tau_vdw = d_tau_cn = nullptr;
  cap_tau_vdw = cap_tau_cn = 0;

  natoms = 0;
  cap_atoms = 0;
  h_x = nullptr;
  h_type = nullptr;
  h_buf = nullptr;
  d_x = nullptr;
  d_type = nullptr;
  d_cn = d_dc6i = d_f = nullptr;
  d_red = nullptr;

  pair_lo = pair_hi = 0;
  device_id = -1;

  const char *tenv = std::getenv("D3_MPI_TIMING");
  timing_flag = (tenv && std::atoi(tenv) > 0) ? 1 : 0;
  t_gather = t_kernel = t_comm = t_total = 0.0;
  ncalls = 0;
}

/* ---------------------------------------------------------------------- */

PairD3MPI::~PairD3MPI()
{
  if (copymode) return;

  if (timing_flag && ncalls > 0)
    fprintf(stdout,
            "d3/mpi rank %d (gpu %d): %ld calls, total %.4f s | gather %.4f s | kernels %.4f s "
            "| comm/wait %.4f s\n",
            comm->me, device_id, ncalls, t_total, t_gather, t_kernel, t_comm);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }

  if (device_id >= 0) {
    cudaFree(d_r2r4);
    cudaFree(d_rcov);
    cudaFree(d_r0ab);
    cudaFree(d_c6ab);
    cudaFree(d_mxc);
    cudaFree(d_tau_vdw);
    cudaFree(d_tau_cn);
    cudaFree(d_x);
    cudaFree(d_type);
    cudaFree(d_cn);
    cudaFree(d_dc6i);
    cudaFree(d_f);
    cudaFree(d_red);
    cudaFreeHost(h_x);
    cudaFreeHost(h_type);
    cudaFreeHost(h_buf);
  }
}

/* ---------------------------------------------------------------------- */

void PairD3MPI::allocate()
{
  allocated = 1;
  const int np1 = atom->ntypes + 1;
  memory->create(setflag, np1, np1, "pair:setflag");
  memory->create(cutsq, np1, np1, "pair:cutsq");
  for (int i = 0; i < np1; i++)
    for (int j = 0; j < np1; j++) setflag[i][j] = 0;
}

/* ----------------------------------------------------------------------
   one GPU per rank: node-local rank modulo visible device count
------------------------------------------------------------------------- */

void PairD3MPI::setup_device()
{
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0)
    error->one(FLERR, "pair d3/mpi: no CUDA device found on rank {}", comm->me);

  int local_rank = 0;
#ifndef MPI_STUBS
  MPI_Comm node_comm;
  MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, comm->me, MPI_INFO_NULL, &node_comm);
  MPI_Comm_rank(node_comm, &local_rank);
  MPI_Comm_free(&node_comm);
#endif

  if (device_id < 0) {
    device_id = local_rank % ndev;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMalloc(&d_red, 10 * sizeof(double)));
  }

  // report the rank -> host/GPU mapping
  char host[MPI_MAX_PROCESSOR_NAME] = "localhost";
#ifndef MPI_STUBS
  int hlen = 0;
  MPI_Get_processor_name(host, &hlen);
#endif
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  std::string line = fmt::format("  rank {:4d}  host {}  gpu {} ({})\n", comm->me, host, device_id,
                                 prop.name);

  const int nprocs = comm->nprocs;
  int len = static_cast<int>(line.size()) + 1;
  std::vector<int> lens(nprocs), offs(nprocs);
  MPI_Gather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, 0, world);
  std::vector<char> all;
  if (comm->me == 0) {
    std::partial_sum(lens.begin(), lens.end() - 1, offs.begin() + 1);
    offs[0] = 0;
    all.resize(offs[nprocs - 1] + lens[nprocs - 1]);
  }
  MPI_Gatherv(const_cast<char *>(line.c_str()), len, MPI_CHAR, all.data(), lens.data(),
              offs.data(), MPI_CHAR, 0, world);
  if (comm->me == 0) {
    std::string msg = fmt::format("pair d3/mpi: {} MPI rank(s), one GPU each\n", nprocs);
    for (int r = 0; r < nprocs; r++) msg += std::string(all.data() + offs[r]);
    utils::logmesg(lmp, msg);
  }
}

/* ----------------------------------------------------------------------
   pair_style d3/mpi rthr cnthr damping functional
------------------------------------------------------------------------- */

void PairD3MPI::settings(int narg, char **arg)
{
  if (narg != 4)
    error->all(FLERR,
               "Pair_style d3/mpi needs four arguments:\n"
               "\t rthr: squared cutoff for dispersion interaction (bohr^2)\n"
               "\t cnthr: squared cutoff for coordination number (bohr^2)\n"
               "\t damping: damp_zero or damp_bj\n"
               "\t functional: name of the functional (e.g., pbe, b3-lyp)");

  rthr = utils::numeric(FLERR, arg[0], false, lmp);
  cnthr = utils::numeric(FLERR, arg[1], false, lmp);

  std::map<std::string, int> damp_map = {
      {"damp_zero", 0}, {"damp_bj", 1}, {"damp_zerom", 2}, {"damp_bjm", 3}};
  if (damp_map.find(arg[2]) == damp_map.end()) error->all(FLERR, "Unknown damping function");
  damping = damp_map[arg[2]];
  if (damping >= 2)
    error->all(FLERR, "pair d3/mpi: {} is not implemented (use damp_zero or damp_bj)", arg[2]);

  functional = arg[3];
  setfuncpar();

  setup_device();
}

/* ---------------------------------------------------------------------- */

int PairD3MPI::find_atomic_number(std::string &key)
{
  std::transform(key.begin(), key.end(), key.begin(), ::tolower);
  if (key.length() == 1) key += " ";
  key.resize(2);

  static const char *element_table[] = {
      "h ", "he", "li", "be", "b ", "c ", "n ", "o ", "f ", "ne", "na", "mg", "al", "si",
      "p ", "s ", "cl", "ar", "k ", "ca", "sc", "ti", "v ", "cr", "mn", "fe", "co", "ni",
      "cu", "zn", "ga", "ge", "as", "se", "br", "kr", "rb", "sr", "y ", "zr", "nb", "mo",
      "tc", "ru", "rh", "pd", "ag", "cd", "in", "sn", "sb", "te", "i ", "xe", "cs", "ba",
      "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb", "dy", "ho", "er", "tm", "yb",
      "lu", "hf", "ta", "w ", "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi", "po",
      "at", "rn", "fr", "ra", "ac", "th", "pa", "u ", "np", "pu"};

  for (int i = 0; i < 94; i++)
    if (key == element_table[i]) return i + 1;
  return -1;
}

/* ----------------------------------------------------------------------
   functional parameters (identical to pair_style d3)
------------------------------------------------------------------------- */

void PairD3MPI::setfuncpar_zero() {
    s6 = 1.0;
    alp = 14.0;
    rs18 = 1.0;

    // default def2-QZVP (almost basis set limit)
    std::unordered_map<std::string, int> commandMap = {
    { "slater-dirac-exchange", 1}, { "b-lyp", 2 },    { "b-p", 3 },       { "b97-d", 4 },      { "revpbe", 5 },
    { "pbe", 6 },                  { "pbesol", 7 },   { "rpw86-pbe", 8 }, { "rpbe", 9 },       { "tpss", 10 },
    { "b3-lyp", 11 },              { "pbe0", 12 },    { "hse06", 13 },    { "revpbe38", 14 },  { "pw6b95", 15 },
    { "tpss0", 16 },               { "b2-plyp", 17 }, { "pwpb95", 18 },   { "b2gp-plyp", 19 }, { "ptpss", 20 },
    { "hf", 21 },                  { "mpwlyp", 22 },  { "bpbe", 23 },     { "bh-lyp", 24 },    { "tpssh", 25 },
    { "pwb6k", 26 },               { "b1b95", 27 },   { "bop", 28 },      { "o-lyp", 29 },     { "o-pbe", 30 },
    { "ssb", 31 },                 { "revssb", 32 },  { "otpss", 33 },    { "b3pw91", 34 },    { "revpbe0", 35 },
    { "pbe38", 36 },               { "mpw1b95", 37 }, { "mpwb1k", 38 },   { "bmk", 39 },       { "cam-b3lyp", 40 },
    { "lc-wpbe", 41 },             { "m05", 42 },     { "m052x", 43 },    { "m06l", 44 },      { "m06", 45 },
    { "m062x", 46 },               { "m06hf", 47 },   { "hcth120", 48 }
    };

    int commandCode = commandMap[functional];
    switch (commandCode) {
    case 1: rs6 = 0.999; s18 = -1.957; rs18 = 0.697; break;
    case 2: rs6 = 1.094; s18 = 1.682; break;
    case 3: rs6 = 1.139; s18 = 1.683; break;
    case 4: rs6 = 0.892; s18 = 0.909; break;
    case 5: rs6 = 0.923; s18 = 1.010; break;
    case 6: rs6 = 1.217; s18 = 0.722; break;
    case 7: rs6 = 1.345; s18 = 0.612; break;
    case 8: rs6 = 1.224; s18 = 0.901; break;
    case 9: rs6 = 0.872; s18 = 0.514; break;
    case 10: rs6 = 1.166; s18 = 1.105; break;
    case 11: rs6 = 1.261; s18 = 1.703; break;
    case 12: rs6 = 1.287; s18 = 0.928; break;
    case 13: rs6 = 1.129; s18 = 0.109; break;
    case 14: rs6 = 1.021; s18 = 0.862; break;
    case 15: rs6 = 1.532; s18 = 0.862; break;
    case 16: rs6 = 1.252; s18 = 1.242; break;
    case 17: rs6 = 1.427; s18 = 1.022; s6 = 0.64; break;
    case 18: rs6 = 1.557; s18 = 0.705; s6 = 0.82; break;
    case 19: rs6 = 1.586; s18 = 0.760; s6 = 0.56; break;
    case 20: rs6 = 1.541; s18 = 0.879; s6 = 0.75; break;
    case 21: rs6 = 1.158; s18 = 1.746; break;
    case 22: rs6 = 1.239; s18 = 1.098; break;
    case 23: rs6 = 1.087; s18 = 2.033; break;
    case 24: rs6 = 1.370; s18 = 1.442; break;
    case 25: rs6 = 1.223; s18 = 1.219; break;
    case 26: rs6 = 1.660; s18 = 0.550; break;
    case 27: rs6 = 1.613; s18 = 1.868; break;
    case 28: rs6 = 0.929; s18 = 1.975; break;
    case 29: rs6 = 0.806; s18 = 1.764; break;
    case 30: rs6 = 0.837; s18 = 2.055; break;
    case 31: rs6 = 1.215; s18 = 0.663; break;
    case 32: rs6 = 1.221; s18 = 0.560; break;
    case 33: rs6 = 1.128; s18 = 1.494; break;
    case 34: rs6 = 1.176; s18 = 1.775; break;
    case 35: rs6 = 0.949; s18 = 0.792; break;
    case 36: rs6 = 1.333; s18 = 0.998; break;
    case 37: rs6 = 1.605; s18 = 1.118; break;
    case 38: rs6 = 1.671; s18 = 1.061; break;
    case 39: rs6 = 1.931; s18 = 2.168; break;
    case 40: rs6 = 1.378; s18 = 1.217; break;
    case 41: rs6 = 1.355; s18 = 1.279; break;
    case 42: rs6 = 1.373; s18 = 0.595; break;
    case 43: rs6 = 1.417; s18 = 0.000; break;
    case 44: rs6 = 1.581; s18 = 0.000; break;
    case 45: rs6 = 1.325; s18 = 0.000; break;
    case 46: rs6 = 1.619; s18 = 0.000; break;
    case 47: rs6 = 1.446; s18 = 0.000; break;
    /* DFTB3(zeta = 4.0), old deprecated parameters; case ("dftb3"); rs6 = 1.235; s18 = 0.673; */
    case 48: rs6 = 1.221; s18 = 1.206; break;
    default:
        error->all(FLERR, "Functional name unknown");
        break;
    }
}

void PairD3MPI::setfuncpar_bj() {
    s6 = 1.0;
    alp = 14.0;

    std::unordered_map<std::string, int> commandMap = {
        {"b-p", 1}, {"b-lyp", 2}, {"revpbe", 3}, {"rpbe", 4}, {"b97-d", 5}, {"pbe", 6},
        {"rpw86-pbe", 7}, {"b3-lyp", 8}, {"tpss", 9}, {"hf", 10}, {"tpss0", 11}, {"pbe0", 12},
        {"hse06", 13}, {"revpbe38", 14}, {"pw6b95", 15}, {"b2-plyp", 16}, {"dsd-blyp", 17},
        {"dsd-blyp-fc", 18}, {"bop", 19}, {"mpwlyp", 20}, {"o-lyp", 21}, {"pbesol", 22}, {"bpbe", 23},
        {"opbe", 24}, {"ssb", 25}, {"revssb", 26}, {"otpss", 27}, {"b3pw91", 28}, {"bh-lyp", 29},
        {"revpbe0", 30}, {"tpssh", 31}, {"mpw1b95", 32}, {"pwb6k", 33}, {"b1b95", 34}, {"bmk", 35},
        {"cam-b3lyp", 36}, {"lc-wpbe", 37}, {"b2gp-plyp", 38}, {"ptpss", 39}, {"pwpb95", 40},
        {"hf/mixed", 41}, {"hf/sv", 42}, {"hf/minis", 43}, {"b3-lyp/6-31gd", 44}, {"hcth120", 45},
        {"pw1pw", 46}, {"pwgga", 47}, {"hsesol", 48}, {"hf3c", 49}, {"hf3cv", 50}, {"pbeh3c", 51},
        {"pbeh-3c", 52}, {"wb97m", 53}, {"r2scan", 54}
    };

    int commandCode = commandMap[functional];
    switch (commandCode) {
        case 1: rs6 = 0.3946; s18 = 3.2822; rs18 = 4.8516; break;
        case 2: rs6 = 0.4298; s18 = 2.6996; rs18 = 4.2359; break;
        case 3: rs6 = 0.5238; s18 = 2.3550; rs18 = 3.5016; break;
        case 4: rs6 = 0.1820; s18 = 0.8318; rs18 = 4.0094; break;
        case 5: rs6 = 0.5545; s18 = 2.2609; rs18 = 3.2297; break;
        case 6: rs6 = 0.4289; s18 = 0.7875; rs18 = 4.4407; break;
        case 7: rs6 = 0.4613; s18 = 1.3845; rs18 = 4.5062; break;
        case 8: rs6 = 0.3981; s18 = 1.9889; rs18 = 4.4211; break;
        case 9: rs6 = 0.4535; s18 = 1.9435; rs18 = 4.4752; break;
        case 10: rs6 = 0.3385; s18 = 0.9171; rs18 = 2.8830; break;
        case 11: rs6 = 0.3768; s18 = 1.2576; rs18 = 4.5865; break;
        case 12: rs6 = 0.4145; s18 = 1.2177; rs18 = 4.8593; break;
        case 13: rs6 = 0.383; s18 = 2.310; rs18 = 5.685; break;
        case 14: rs6 = 0.4309; s18 = 1.4760; rs18 = 3.9446; break;
        case 15: rs6 = 0.2076; s18 = 0.7257; rs18 = 6.3750; break;
        case 16: rs6 = 0.3065; s18 = 0.9147; rs18 = 5.0570; s6 = 0.64; break; // s6 was dead code in pair d3
        case 17: rs6 = 0.0000; s18 = 0.2130; rs18 = 6.0519; s6 = 0.50; break;
        case 18: rs6 = 0.0009; s18 = 0.2112; rs18 = 5.9807; s6 = 0.50; break;
        case 19: rs6 = 0.4870; s18 = 3.2950; rs18 = 3.5043; break;
        case 20: rs6 = 0.4831; s18 = 2.0077; rs18 = 4.5323; break;
        case 21: rs6 = 0.5299; s18 = 2.6205; rs18 = 2.8065; break;
        case 22: rs6 = 0.4466; s18 = 2.9491; rs18 = 6.1742; break;
        case 23: rs6 = 0.4567; s18 = 4.0728; rs18 = 4.3908; break;
        case 24: rs6 = 0.5512; s18 = 3.3816; rs18 = 2.9444; break;
        case 25: rs6 = -0.0952; s18 = -0.1744; rs18 = 5.2170; break;
        case 26: rs6 = 0.4720; s18 = 0.4389; rs18 = 4.0986; break;
        case 27: rs6 = 0.4634; s18 = 2.7495; rs18 = 4.3153; break;
        case 28: rs6 = 0.4312; s18 = 2.8524; rs18 = 4.4693; break;
        case 29: rs6 = 0.2793; s18 = 1.0354; rs18 = 4.9615; break;
        case 30: rs6 = 0.4679; s18 = 1.7588; rs18 = 3.7619; break;
        case 31: rs6 = 0.4529; s18 = 2.2382; rs18 = 4.6550; break;
        case 32: rs6 = 0.1955; s18 = 1.0508; rs18 = 6.4177; break;
        case 33: rs6 = 0.1805; s18 = 0.9383; rs18 = 7.7627; break;
        case 34: rs6 = 0.2092; s18 = 1.4507; rs18 = 5.5545; break;
        case 35: rs6 = 0.1940; s18 = 2.0860; rs18 = 5.9197; break;
        case 36: rs6 = 0.3708; s18 = 2.0674; rs18 = 5.4743; break;
        case 37: rs6 = 0.3919; s18 = 1.8541; rs18 = 5.0897; break;
        case 38: rs6 = 0.0000; s18 = 0.2597; rs18 = 6.3332; s6 = 0.560; break;
        case 39: rs6 = 0.0000; s18 = 0.2804; rs18 = 6.5745; s6 = 0.750; break;
        case 40: rs6 = 0.0000; s18 = 0.2904; rs18 = 7.3141; s6 = 0.820; break;
        // special HF / DFT with eBSSE correction;
        case 41: rs6 = 0.5607; s18 = 3.9027; rs18 = 4.5622; break;
        case 42: rs6 = 0.4249; s18 = 2.1849; rs18 = 4.2783; break;
        case 43: rs6 = 0.1702; s18 = 0.9841; rs18 = 3.8506; break;
        case 44: rs6 = 0.5014; s18 = 4.0672; rs18 = 4.8409; break;
        case 45: rs6 = 0.3563; s18 = 1.0821; rs18 = 4.3359; break;
        /*     DFTB3 old, deprecated parameters : ;
            *     case ("dftb3"); rs6 = 0.7461; s18 = 3.209; rs18 = 4.1906;
            *     special SCC - DFTB parametrization;
            *     full third order DFTB, self consistent charges, hydrogen pair damping with; exponent 4.2;
        */
        case 46: rs6 = 0.3807; s18 = 2.3363; rs18 = 5.8844; break;
        case 47: rs6 = 0.2211; s18 = 2.6910; rs18 = 6.7278; break;
        case 48: rs6 = 0.4650; s18 = 2.9215; rs18 = 6.2003; break;
        // special HF - D3 - gCP - SRB / MINIX parametrization;
        case 49: rs6 = 0.4171; s18 = 0.8777; rs18 = 2.9149; break;
        // special HF - D3 - gCP - SRB2 / ECP - 2G parametrization;
        case 50: rs6 = 0.3063; s18 = 0.5022; rs18 = 3.9856; break;
        // special PBEh - D3 - gCP / def2 - mSVP parametrization;
        case 51: rs6 = 0.4860; s18 = 0.0000; rs18 = 4.5000; break;
        case 52: rs6 = 0.4860; s18 = 0.0000; rs18 = 4.5000; break;
        case 53: rs6 = 0.5660; s18 = 0.3908; rs18 = 3.1280; break;  // J. Chem. Theory Comput. 14, 5725 (2018)
        case 54: rs6 = 0.4948; s18 = 0.7898; rs18 = 5.7308; break;  // S. Ehlert et al.,  J. Chem. Phys. 154, 061101 (2021)
        default:
            error->all(FLERR, "Functional name unknown");
            break;
    }
}

void PairD3MPI::setfuncpar_zerom() {
    s6 = 1.0;
    alp = 14.0;

    std::unordered_map<std::string, int> commandMap = {
        {"b2-plyp", 1}, {"b3-lyp", 2}, {"b97-d", 3}, {"b-lyp", 4},
        {"b-p", 5}, {"pbe", 6}, {"pbe0", 7}, {"lc-wpbe", 8}
    };

    int commandCode = commandMap[functional];
    switch (commandCode) {
        case 1: rs6 = 1.313134; s18 = 0.717543; rs18 = 0.016035; s6 = 0.640000; break;
        case 2: rs6 = 1.338153; s18 = 1.532981; rs18 = 0.013988; break;
        case 3: rs6 = 1.151808; s18 = 1.020078; rs18 = 0.035964; break;
        case 4: rs6 = 1.279637; s18 = 1.841686; rs18 = 0.014370; break;
        case 5: rs6 = 1.233460; s18 = 1.945174; rs18 = 0.000000; break;
        case 6: rs6 = 2.340218; s18 = 0.000000; rs18 = 0.129434; break;
        case 7: rs6 = 2.077949; s18 = 0.000081; rs18 = 0.116755; break;
        case 8: rs6 = 1.366361; s18 = 1.280619; rs18 = 0.003160; break;
        default:
            error->all(FLERR, "Functional name unknown");
            break;
    }
}

void PairD3MPI::setfuncpar_bjm() {
    s6 = 1.0;
    alp = 14.0;

    std::unordered_map<std::string, int> commandMap = {
        {"b2-plyp", 1}, {"b3-lyp", 2}, {"b97-d", 3}, {"b-lyp", 4},
        {"b-p", 5}, {"pbe", 6}, {"pbe0", 7}, {"lc-wpbe", 8}
    };

    int commandCode = commandMap[functional];
    switch (commandCode) {
        case 1: rs6 = 0.486434; s18 = 0.672820; rs18 = 3.656466; s6 = 0.640000; break;
        case 2: rs6 = 0.278672; s18 = 1.466677; rs18 = 4.606311; break;
        case 3: rs6 = 0.240184; s18 = 1.206988; rs18 = 3.864426; break;
        case 4: rs6 = 0.448486; s18 = 1.875007; rs18 = 3.610679; break;
        case 5: rs6 = 0.821850; s18 = 3.140281; rs18 = 2.728151; break;
        case 6: rs6 = 0.012092; s18 = 0.358940; rs18 = 5.938951; break;
        case 7: rs6 = 0.007912; s18 = 0.528823; rs18 = 6.162326; break;
        case 8: rs6 = 0.563761; s18 = 0.906564; rs18 = 3.593680; break;
        default:
            error->all(FLERR, "Functional name unknown");
            break;
    }
}

void PairD3MPI::setfuncpar() {
    void (PairD3MPI::*setfuncpar_damp[4])() = {
        &PairD3MPI::setfuncpar_zero,
        &PairD3MPI::setfuncpar_bj,
        &PairD3MPI::setfuncpar_zerom,
        &PairD3MPI::setfuncpar_bjm
    };
    (this->*setfuncpar_damp[damping])();

    rs8 = rs18;
    alp6 = alp;
    alp8 = alp + 2.0;
    // rs10 = rs18
    // alp10 = alp + 4.0;

    a1 = rs6;
    a2 = rs8;
    s8 = s18;
    // s6 is already defined
}

/* ----------------------------------------------------------------------
   pair_coeff * * element1 element2 ...
------------------------------------------------------------------------- */

void PairD3MPI::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  const int ntypes = atom->ntypes;
  const int np1 = ntypes + 1;
  if (narg != ntypes + 2) error->all(FLERR, "Pair_coeff needs: * * element1 element2 ...");
  if (!utils::strmatch(arg[0], "^\\*$") || !utils::strmatch(arg[1], "^\\*$"))
    error->all(FLERR, "Pair_coeff for d3/mpi must start with * *");

  std::vector<int> znum(np1, 0);
  for (int i = 1; i <= ntypes; i++) {
    std::string element = arg[i + 1];
    znum[i] = find_atomic_number(element);
    if (znum[i] < 1) error->all(FLERR, "pair d3/mpi: unknown element {}", arg[i + 1]);
  }

  for (int i = 1; i <= ntypes; i++)
    for (int j = 1; j <= ntypes; j++) setflag[i][j] = 1;

  h_r2r4.assign(np1, 0.0f);
  h_rcov.assign(np1, 0.0f);
  for (int i = 1; i <= ntypes; i++) {
    h_r2r4[i] = r2r4_ref[znum[i] - 1];
    h_rcov[i] = rcov_ref[znum[i] - 1];
  }

  static const double r0ab_table[94][94] = R0AB_TABLE;
  h_r0ab.assign(np1 * np1, 0.0f);
  for (int i = 1; i <= ntypes; i++)
    for (int j = 1; j <= ntypes; j++)
      h_r0ab[i * np1 + j] = r0ab_table[znum[i] - 1][znum[j] - 1] / AU_TO_ANG;

  // C6 reference grid; every atom type mapped to an element gets the entries
  // (several atom types may map to the same element)
  static const double c6ab_table[32385][5] = C6AB_TABLE;
  h_mxc.assign(np1, 0);
  h_c6ab.assign(static_cast<size_t>(np1) * np1 * MAXC * MAXC * 3, -1.0f);
  auto c6idx = [np1](int ti, int tj, int a, int b, int k) {
    return ((((static_cast<size_t>(ti) * np1 + tj) * MAXC + a) * MAXC + b) * 3 + k);
  };

  std::vector<std::vector<int>> types_of_z(95);
  for (int i = 1; i <= ntypes; i++) types_of_z[znum[i]].push_back(i);

  for (int n = 0; n < 32385; n++) {
    const double ref_c6 = c6ab_table[n][0];
    int z1 = static_cast<int>(c6ab_table[n][1]);
    int z2 = static_cast<int>(c6ab_table[n][2]);
    const int gi = (z1 - 1) / 100 + 1;
    const int gj = (z2 - 1) / 100 + 1;
    z1 = (z1 - 1) % 100 + 1;
    z2 = (z2 - 1) % 100 + 1;
    if (z1 > 94 || z2 > 94) continue;
    const double ref_cn1 = c6ab_table[n][3];
    const double ref_cn2 = c6ab_table[n][4];

    for (int ti : types_of_z[z1]) {
      for (int tj : types_of_z[z2]) {
        h_mxc[ti] = std::max(h_mxc[ti], gi);
        h_mxc[tj] = std::max(h_mxc[tj], gj);
        h_c6ab[c6idx(ti, tj, gi - 1, gj - 1, 0)] = ref_c6;
        h_c6ab[c6idx(ti, tj, gi - 1, gj - 1, 1)] = ref_cn1;
        h_c6ab[c6idx(ti, tj, gi - 1, gj - 1, 2)] = ref_cn2;
        h_c6ab[c6idx(tj, ti, gj - 1, gi - 1, 0)] = ref_c6;
        h_c6ab[c6idx(tj, ti, gj - 1, gi - 1, 1)] = ref_cn2;
        h_c6ab[c6idx(tj, ti, gj - 1, gi - 1, 2)] = ref_cn1;
      }
    }
  }

  cudaFree(d_r2r4);
  cudaFree(d_rcov);
  cudaFree(d_r0ab);
  cudaFree(d_c6ab);
  cudaFree(d_mxc);
  CUDA_CHECK(cudaMalloc(&d_r2r4, h_r2r4.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_rcov, h_rcov.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r0ab, h_r0ab.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c6ab, h_c6ab.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mxc, h_mxc.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_r2r4, h_r2r4.data(), h_r2r4.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rcov, h_rcov.data(), h_rcov.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_r0ab, h_r0ab.data(), h_r0ab.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_c6ab, h_c6ab.data(), h_c6ab.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mxc, h_mxc.data(), h_mxc.size() * sizeof(int), cudaMemcpyHostToDevice));
}

/* ---------------------------------------------------------------------- */

double PairD3MPI::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  // all-pairs evaluation over periodic images: no neighbor list / ghost atoms needed
  return 0.0;
}

/* ---------------------------------------------------------------------- */

void PairD3MPI::init_style()
{
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style d3/mpi requires atom IDs");
  if (atom->natoms > MAXSMALLINT) error->all(FLERR, "Too many atoms for pair style d3/mpi");
}

/* ----------------------------------------------------------------------
   (re)allocate per-atom buffers
------------------------------------------------------------------------- */

void PairD3MPI::grow_atom_arrays(int n)
{
  if (n <= cap_atoms) return;
  cudaFree(d_x);
  cudaFree(d_type);
  cudaFree(d_cn);
  cudaFree(d_dc6i);
  cudaFree(d_f);
  cudaFreeHost(h_x);
  cudaFreeHost(h_type);
  cudaFreeHost(h_buf);

  cap_atoms = n;
  CUDA_CHECK(cudaMalloc(&d_x, 3 * static_cast<size_t>(n) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_type, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cn, static_cast<size_t>(n) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dc6i, static_cast<size_t>(n) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_f, 3 * static_cast<size_t>(n) * sizeof(double)));
  CUDA_CHECK(cudaMallocHost(&h_x, 3 * static_cast<size_t>(n) * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_type, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMallocHost(&h_buf, 3 * static_cast<size_t>(n) * sizeof(double)));
}

/* ----------------------------------------------------------------------
   replicate all atoms on every rank; global order = sorted by atom ID
------------------------------------------------------------------------- */

void PairD3MPI::gather_atoms()
{
  const int nprocs = comm->nprocs;
  int nlocal = atom->nlocal;

  counts.resize(nprocs);
  displs.resize(nprocs);
  counts3.resize(nprocs);
  std::vector<int> displs3(nprocs);

  MPI_Allgather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, world);
  long long ntot = 0;
  for (int r = 0; r < nprocs; r++) {
    displs[r] = static_cast<int>(ntot);
    counts3[r] = 3 * counts[r];
    displs3[r] = 3 * displs[r];
    ntot += counts[r];
  }
  if (ntot != atom->natoms) error->all(FLERR, "pair d3/mpi: inconsistent atom count");
  natoms = static_cast<int>(ntot);

  tag_all.resize(natoms);
  type_all.resize(natoms);
  x_all.resize(3 * static_cast<size_t>(natoms));
  gpos.resize(natoms);

  const tagint *tagp = nlocal ? atom->tag : nullptr;
  const int *typep = nlocal ? atom->type : nullptr;
  const double *xp = nlocal ? &atom->x[0][0] : nullptr;

  MPI_Allgatherv(const_cast<tagint *>(tagp), nlocal, MPI_LMP_TAGINT, tag_all.data(),
                 counts.data(), displs.data(), MPI_LMP_TAGINT, world);
  MPI_Allgatherv(const_cast<int *>(typep), nlocal, MPI_INT, type_all.data(), counts.data(),
                 displs.data(), MPI_INT, world);
  MPI_Allgatherv(const_cast<double *>(xp), 3 * nlocal, MPI_DOUBLE, x_all.data(), counts3.data(),
                 displs3.data(), MPI_DOUBLE, world);

  // gathered index -> position in ID-sorted order
  tagint maxtag = 0;
  for (int k = 0; k < natoms; k++) maxtag = std::max(maxtag, tag_all[k]);
  if (maxtag == natoms) {
    for (int k = 0; k < natoms; k++) gpos[k] = static_cast<int>(tag_all[k] - 1);
  } else {
    std::vector<int> order(natoms);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [this](int a, int b) { return tag_all[a] < tag_all[b]; });
    for (int s = 0; s < natoms; s++) gpos[order[s]] = s;
  }

  grow_atom_arrays(natoms);

  // this rank's slice of the pair list
  const long long npairs = static_cast<long long>(natoms) * (natoms + 1) / 2;
  pair_lo = npairs * comm->me / nprocs;
  pair_hi = npairs * (comm->me + 1) / nprocs;
}

/* ----------------------------------------------------------------------
   number of cell repetitions needed along each lattice vector
------------------------------------------------------------------------- */

void PairD3MPI::set_repetition(double r2_threshold, int *rep)
{
  const double rc = sqrt(r2_threshold);
  const double v[3][3] = {{lat[0][0], lat[1][0], lat[2][0]},
                          {lat[0][1], lat[1][1], lat[2][1]},
                          {lat[0][2], lat[1][2], lat[2][2]}};
  auto cross = [](const double *a, const double *b, double *c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
  };
  auto dot = [](const double *a, const double *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  };
  double c12[3], c23[3], c31[3];
  cross(v[0], v[1], c12);
  cross(v[1], v[2], c23);
  cross(v[2], v[0], c31);

  rep[0] = static_cast<int>(std::abs(rc / (dot(c23, v[0]) / sqrt(dot(c23, c23))))) + 1;
  rep[1] = static_cast<int>(std::abs(rc / (dot(c31, v[1]) / sqrt(dot(c31, c31))))) + 1;
  rep[2] = static_cast<int>(std::abs(rc / (dot(c12, v[2]) / sqrt(dot(c12, c12))))) + 1;

  if (domain->xperiodic == 0) rep[0] = 0;
  if (domain->yperiodic == 0) rep[1] = 0;
  if (domain->zperiodic == 0) rep[2] = 0;
}

/* ----------------------------------------------------------------------
   cell translation vectors, ordered (x outer, z inner) as in pair d3
------------------------------------------------------------------------- */

void PairD3MPI::build_tau(const int *rep, std::vector<float> &tau)
{
  tau.clear();
  for (int tx = -rep[0]; tx <= rep[0]; tx++)
    for (int ty = -rep[1]; ty <= rep[1]; ty++)
      for (int tz = -rep[2]; tz <= rep[2]; tz++)
        for (int d = 0; d < 3; d++)
          tau.push_back(static_cast<float>(lat[d][0] * tx + lat[d][1] * ty + lat[d][2] * tz));
}

/* ----------------------------------------------------------------------
   lattice, periodic images, and wrapped positions (bohr)
------------------------------------------------------------------------- */

void PairD3MPI::set_lattice()
{
  lat[0][0] = domain->xprd / AU_TO_ANG;
  lat[1][0] = 0.0;
  lat[2][0] = 0.0;
  lat[0][1] = domain->xy / AU_TO_ANG;
  lat[1][1] = domain->yprd / AU_TO_ANG;
  lat[2][1] = 0.0;
  lat[0][2] = domain->xz / AU_TO_ANG;
  lat[1][2] = domain->yz / AU_TO_ANG;
  lat[2][2] = domain->zprd / AU_TO_ANG;

  set_repetition(rthr, rep_vdw);
  set_repetition(cnthr, rep_cn);

  std::vector<float> tau;
  build_tau(rep_vdw, tau);
  nimg_vdw = static_cast<int>(tau.size() / 3);
  if (nimg_vdw > cap_tau_vdw) {
    cudaFree(d_tau_vdw);
    CUDA_CHECK(cudaMalloc(&d_tau_vdw, tau.size() * sizeof(float)));
    cap_tau_vdw = nimg_vdw;
  }
  CUDA_CHECK(cudaMemcpy(d_tau_vdw, tau.data(), tau.size() * sizeof(float), cudaMemcpyHostToDevice));

  build_tau(rep_cn, tau);
  nimg_cn = static_cast<int>(tau.size() / 3);
  if (nimg_cn > cap_tau_cn) {
    cudaFree(d_tau_cn);
    CUDA_CHECK(cudaMalloc(&d_tau_cn, tau.size() * sizeof(float)));
    cap_tau_cn = nimg_cn;
  }
  CUDA_CHECK(cudaMemcpy(d_tau_cn, tau.data(), tau.size() * sizeof(float), cudaMemcpyHostToDevice));

  // lat is upper triangular
  double inv[3][3] = {{0.0}};
  inv[0][0] = 1.0 / lat[0][0];
  inv[1][1] = 1.0 / lat[1][1];
  inv[2][2] = 1.0 / lat[2][2];
  inv[0][1] = -lat[0][1] / (lat[0][0] * lat[1][1]);
  inv[1][2] = -lat[1][2] / (lat[1][1] * lat[2][2]);
  inv[0][2] = (lat[0][1] * lat[1][2] - lat[0][2] * lat[1][1]) / (lat[0][0] * lat[1][1] * lat[2][2]);

  const int periodic[3] = {domain->xperiodic, domain->yperiodic, domain->zperiodic};
  const double *boxlo = domain->boxlo;

  for (int k = 0; k < natoms; k++) {
    const double r[3] = {(x_all[3 * k] - boxlo[0]) / AU_TO_ANG,
                         (x_all[3 * k + 1] - boxlo[1]) / AU_TO_ANG,
                         (x_all[3 * k + 2] - boxlo[2]) / AU_TO_ANG};
    double a[3];
    for (int i = 0; i < 3; i++) {
      a[i] = inv[i][0] * r[0] + inv[i][1] * r[1] + inv[i][2] * r[2];
      if (periodic[i]) a[i] -= floor(a[i]);
    }
    const int g = gpos[k];
    for (int i = 0; i < 3; i++)
      h_x[3 * g + i] = static_cast<float>(lat[i][0] * a[0] + lat[i][1] * a[1] + lat[i][2] * a[2]);
    h_type[g] = type_all[k];
  }
}

/* ----------------------------------------------------------------------
   in-place sum of a device array over all ranks (host staged)
------------------------------------------------------------------------- */

void PairD3MPI::allreduce_device(double *dptr, int n)
{
  if (comm->nprocs == 1) return;
  CUDA_CHECK(cudaMemcpy(h_buf, dptr, n * sizeof(double), cudaMemcpyDeviceToHost));
  MPI_Allreduce(MPI_IN_PLACE, h_buf, n, MPI_DOUBLE, MPI_SUM, world);
  CUDA_CHECK(cudaMemcpy(dptr, h_buf, n * sizeof(double), cudaMemcpyHostToDevice));
}

/* ----------------------------------------------------------------------
   energy, forces, virial
------------------------------------------------------------------------- */

void PairD3MPI::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  static bool warned_peratom = false;
  if ((eflag_atom || vflag_atom) && !warned_peratom) {
    if (comm->me == 0)
      error->warning(FLERR, "pair d3/mpi does not compute per-atom energy or virial");
    warned_peratom = true;
  }

  double t0 = MPI_Wtime();

  gather_atoms();
  if (natoms == 0) return;
  set_lattice();

  CUDA_CHECK(cudaMemcpy(d_x, h_x, 3 * static_cast<size_t>(natoms) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_type, h_type, static_cast<size_t>(natoms) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_cn, 0, static_cast<size_t>(natoms) * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_dc6i, 0, static_cast<size_t>(natoms) * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_f, 0, 3 * static_cast<size_t>(natoms) * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_red, 0, 10 * sizeof(double)));

  const long long npairs = pair_hi - pair_lo;
  const int blocks = launch_blocks(npairs);
  const int self_vdw = (rep_vdw[0] * (2 * rep_vdw[1] + 1) + rep_vdw[1]) * (2 * rep_vdw[2] + 1) +
      rep_vdw[2];
  const int self_cn =
      (rep_cn[0] * (2 * rep_cn[1] + 1) + rep_cn[1]) * (2 * rep_cn[2] + 1) + rep_cn[2];
  const int np1 = atom->ntypes + 1;

  double t1 = MPI_Wtime();
  double tk = 0.0, tc = 0.0, ts;

  // 1. coordination numbers
  ts = MPI_Wtime();
  if (npairs > 0)
    kernel_cn<<<blocks, D3_THREADS>>>(pair_lo, pair_hi, nimg_cn, self_cn, cnthr, K1, d_rcov,
                                      d_tau_cn, d_type, d_x, d_cn);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  tk += MPI_Wtime() - ts;
  ts = MPI_Wtime();
  allreduce_device(d_cn, natoms);
  tc += MPI_Wtime() - ts;

  // 2. C6(CN), energy, forces at fixed C6, dE/dCN
  ts = MPI_Wtime();
  if (npairs > 0) {
    if (damping == 0)
      kernel_vdw<0><<<blocks, D3_THREADS>>>(pair_lo, pair_hi, nimg_vdw, self_vdw, np1, rthr, s6, s8,
                                            a1, a2, alp6, alp8, K3, d_r2r4, d_r0ab, d_mxc, d_c6ab,
                                            d_tau_vdw, d_type, d_x, d_cn, d_dc6i, d_f, d_red);
    else
      kernel_vdw<1><<<blocks, D3_THREADS>>>(pair_lo, pair_hi, nimg_vdw, self_vdw, np1, rthr, s6, s8,
                                            a1, a2, alp6, alp8, K3, d_r2r4, d_r0ab, d_mxc, d_c6ab,
                                            d_tau_vdw, d_type, d_x, d_cn, d_dc6i, d_f, d_red);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  tk += MPI_Wtime() - ts;
  ts = MPI_Wtime();
  allreduce_device(d_dc6i, natoms);
  tc += MPI_Wtime() - ts;

  // 3. forces from dC6/dCN
  ts = MPI_Wtime();
  if (npairs > 0)
    kernel_dc6<<<blocks, D3_THREADS>>>(pair_lo, pair_hi, nimg_cn, self_cn, cnthr, K1, d_dc6i,
                                       d_rcov, d_tau_cn, d_type, d_x, d_f, d_red);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  tk += MPI_Wtime() - ts;

  // 4. forces back to the owning ranks
  ts = MPI_Wtime();
  CUDA_CHECK(cudaMemcpy(h_buf, d_f, 3 * static_cast<size_t>(natoms) * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_red, d_red, 10 * sizeof(double), cudaMemcpyDeviceToHost));

  const int nlocal = atom->nlocal;
  f_local.resize(3 * static_cast<size_t>(nlocal));
  if (comm->nprocs == 1) {
    for (int i = 0; i < nlocal; i++)
      for (int d = 0; d < 3; d++) f_local[3 * i + d] = h_buf[3 * gpos[i] + d];
  } else {
    f_gather.resize(3 * static_cast<size_t>(natoms));
    for (int k = 0; k < natoms; k++)
      for (int d = 0; d < 3; d++) f_gather[3 * k + d] = h_buf[3 * gpos[k] + d];
    MPI_Reduce_scatter(f_gather.data(), f_local.data(), counts3.data(), MPI_DOUBLE, MPI_SUM,
                       world);
  }
  tc += MPI_Wtime() - ts;

  const double fconv = AU_TO_EV / AU_TO_ANG;
  double **f = atom->f;
  for (int i = 0; i < nlocal; i++)
    for (int d = 0; d < 3; d++) f[i][d] += f_local[3 * i + d] * fconv;

  // per-rank partial sums; LAMMPS sums eng_vdwl and virial over ranks
  if (eflag_global) eng_vdwl += h_red[9] * AU_TO_EV;
  if (vflag_global) {
    virial[0] += h_red[0] * AU_TO_EV;
    virial[1] += h_red[4] * AU_TO_EV;
    virial[2] += h_red[8] * AU_TO_EV;
    virial[3] += h_red[1] * AU_TO_EV;
    virial[4] += h_red[2] * AU_TO_EV;
    virial[5] += h_red[5] * AU_TO_EV;
  }

  if (timing_flag) {
    const double t2 = MPI_Wtime();
    t_gather += t1 - t0;
    t_kernel += tk;
    t_comm += tc;
    t_total += t2 - t0;
    ncalls++;
  }
}
