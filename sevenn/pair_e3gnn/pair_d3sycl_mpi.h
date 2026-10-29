/* -*- c++ -*- ----------------------------------------------------------
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
   Multi-GPU (MPI) SYCL variant of pair_style d3 -- pair style "d3sycl/mpi".
   SYCL port of pair_d3_mpi.cu (lammps/src); same algorithm:
   every rank holds a replica of all atom positions (sorted by atom ID)
   and evaluates a contiguous slice of the N(N+1)/2 atom-pair list on its
   own SYCL device. Per-atom CN, dE/dCN and forces are combined with MPI.

   Original CUDA D3 implementation: Gijin Kim, Hyungmin An (SNU)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(d3sycl/mpi,PairD3SyclMPI);
// clang-format on
#else

#ifndef LMP_PAIR_D3SYCL_MPI_H
#define LMP_PAIR_D3SYCL_MPI_H

#include "pair.h"

#include <string>
#include <vector>

// <sycl/sycl.hpp> is kept out of this header: it is also included by
// host-compiler translation units (force.cpp via style_pair.h).

namespace LAMMPS_NS {

class PairD3SyclMPI : public Pair {
 public:
  PairD3SyclMPI(class LAMMPS *);
  ~PairD3SyclMPI() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;

  static constexpr int MAXC = 5;    // maximum coordination number references per element

 protected:
  void allocate();
  void setup_device();
  void upload_params();
  void upload_tau(const std::vector<float> &, float *&, int &);

  int find_atomic_number(std::string &);
  void setfuncpar_zero();
  void setfuncpar_bj();
  void setfuncpar_zerom();
  void setfuncpar_bjm();
  void setfuncpar();

  void gather_atoms();
  void set_lattice();
  void set_repetition(double, int *);
  void build_tau(const int *, std::vector<float> &);
  void grow_atom_arrays(int);
  void allreduce_device(double *, int);
  void free_device_memory();

  static constexpr double AU_TO_ANG = 0.52917726;
  static constexpr double AU_TO_EV = 27.21138505;
  static constexpr float K1 = 16.0f;
  static constexpr float K3 = -4.0f;

  // parameters
  int damping;
  std::string functional;
  float rthr, cnthr;
  float s6, s8, s18, rs6, rs8, rs18, alp, alp6, alp8, a1, a2;

  // host copies of element parameters (index = atom type, 1..ntypes)
  std::vector<float> h_r2r4, h_rcov, h_r0ab, h_c6ab;
  std::vector<int> h_mxc;

  // device copies of element parameters
  float *d_r2r4, *d_rcov, *d_r0ab, *d_c6ab;
  int *d_mxc;

  // lattice
  double lat[3][3];    // columns are the lattice vectors (bohr)
  int rep_vdw[3], rep_cn[3];
  int nimg_vdw, nimg_cn;
  float *d_tau_vdw, *d_tau_cn;
  int cap_tau_vdw, cap_tau_cn;

  // global (replicated) atom data, ordered by atom ID
  int natoms;
  int cap_atoms;
  std::vector<int> counts, displs, counts3;    // per-rank local atom counts (gather layout)
  std::vector<int> gpos;                       // gathered index -> tag-sorted index
  std::vector<tagint> tag_all;
  std::vector<int> type_all;
  std::vector<double> x_all;
  std::vector<double> f_gather;    // global forces in gathered (rank-major) order
  std::vector<double> f_local;

  float *h_x;        // USM host (pinned)
  int *h_type;       // USM host (pinned)
  double *h_buf;     // USM host (pinned), 3*natoms
  float *d_x;
  int *d_type;
  double *d_cn, *d_dc6i, *d_f;
  double *d_red;    // sigma[9] + disp
  double h_red[10];

  // this rank's slice of the pair list
  long long pair_lo, pair_hi;

  // SYCL device / timing
  void *qptr;    // sycl::queue *
  int device_id;
  std::string device_name;
  int timing_flag;
  double t_gather, t_kernel, t_comm, t_total;
  long ncalls;
};

}    // namespace LAMMPS_NS

#endif
#endif
