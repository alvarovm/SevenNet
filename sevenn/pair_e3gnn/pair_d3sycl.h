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
   SYCL port of pair_d3.cu (CUDA) -- pair style "d3sycl"
   Original contributing authors (CUDA version): Gijin Kim, Hyungmin An (SNU)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
PairStyle(d3sycl, PairD3Sycl)

#else

#ifndef LMP_PAIR_D3SYCL
#define LMP_PAIR_D3SYCL

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <sycl/sycl.hpp>

#include "pair.h"
#include "utils.h"
#include "atom.h"
#include "domain.h"
#include "error.h"
#include "comm.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "math_extra.h"

#include "pair_d3_pars.h"

namespace LAMMPS_NS {
    class PairD3Sycl : public Pair {
    public:
        PairD3Sycl(class LAMMPS*);
        ~PairD3Sycl() override;

        void compute(int, int) override;
        void settings(int, char**) override;
        void coeff(int, char**) override;
        double init_one(int i, int j) override;
        void init_style() override;

        void write_restart(FILE*) override;
        void read_restart(FILE*) override;
        void write_restart_settings(FILE*) override;
        void read_restart_settings(FILE*) override;

    protected:
        virtual void allocate();

        /* ------- Read parameters ------- */
        int find_atomic_number(std::string&);
        int is_int_in_array(int*, int, int);
        void read_r0ab(int*, int);
        void get_limit_in_pars_array(int&, int&, int&, int&);
        void read_c6ab(int*, int);

        void setfuncpar_zero();
        void setfuncpar_bj();
        void setfuncpar_zerom();
        void setfuncpar_bjm();
        void setfuncpar();
        /* ------- Read parameters ------- */

        /* ------- Lattice information ------- */
        void set_lattice_repetition_criteria(float, int*);
        void set_lattice_vectors();
        /* ------- Lattice information ------- */

        /* ------- Initialize & Precalculate ------- */
        void load_atom_info();
        void precalculate_tau_array();
        /* ------- Initialize & Precalculate ------- */

        /* ------- Reallocate (when number of atoms changed) ------- */
        void reallocate_arrays();
        /* ------- Reallocate (when number of atoms changed) ------- */

        /* ------- Coordination number ------- */
        void get_coordination_number();
        void get_dC6_dCNij();
        /* ------- Coordination number ------- */

        /* ------- Main workers ------- */
        void get_forces_without_dC6_zero();
        void get_forces_without_dC6_bj();
        void get_forces_without_dC6_zerom();
        void get_forces_without_dC6_bjm();
        void get_forces_without_dC6();
        void get_forces_with_dC6();
        void update(int, int);
        /* ------- Main workers ------- */

        /* ------- USM allocation helpers ------- */
        template <typename T> T* usm_alloc(size_t count) {
            return sycl::malloc_shared<T>(count, queue);
        }
        template <typename T> void usm_free(T*& ptr) {
            if (ptr) { sycl::free(ptr, queue); ptr = nullptr; }
        }
        /* ------- USM allocation helpers ------- */

        /*--------- Constants ---------*/
        static constexpr int MAX_ELEM = 94;          // maximum of the element number
        static constexpr int MAXC = 5;               // maximum coordination number references per element

        static constexpr double AU_TO_ANG = 0.52917726; // conversion factors (atomic unit --> angstrom)
        static constexpr double AU_TO_EV = 27.21138505; // conversion factors (atomic unit --> eV)

        static constexpr float K1 = 16.0;              // global ad hoc parameters
        static constexpr float K3 = -4.0;              // global ad hoc parameters
        /*--------- Constants ---------*/

        /*--------- SYCL queue ---------*/
        sycl::queue queue;
        /*--------- SYCL queue ---------*/

        /*--------- Parameters to read ---------*/
        int damping;
        std::string functional;
        float* r2r4 = nullptr;             // scale r4/r2 values of the atoms by sqrt(Z)        [np1]
        float* rcov = nullptr;             // covalent radii                                    [np1]
        int* mxc = nullptr;                // How large the grid for c6 interpolation           [np1]
        float* r0ab = nullptr;             // cut-off radii for all element pairs       [np1*np1]
        float* c6ab = nullptr;             // C6 for all element pairs      [np1*np1*MAXC*MAXC*3]
        int np1_save = 0;                  // ntypes+1 at allocation time (for c6ab/r0ab indexing)
        float rthr;                        // R^2 distance to cutoff for C calculation
        float cnthr;                       // R^2 distance to cutoff for CN_calculation
        float s6, s8, s18, rs6, rs8, rs18, alp, alp6, alp8, a1, a2; // parameters for D3
        /*--------- Parameters to read ---------*/

        /*--------- Lattice related values ---------*/
        double lat_v_1[3] = {0.0, 0.0, 0.0}; // lattice coordination vector
        double lat_v_2[3] = {0.0, 0.0, 0.0}; // lattice coordination vector
        double lat_v_3[3] = {0.0, 0.0, 0.0}; // lattice coordination vector
        int* rep_vdw = nullptr;              // repetition of cell for calculating D3 (USM, size 3)
        int* rep_cn = nullptr;               // repetition of cell for calculating CN (USM, size 3)
        double* sigma = nullptr;             // virial pressure on cell, flattened 3x3 (USM, size 9)
        /*--------- Lattice related values ---------*/

        /*--------- Per-atom values/arrays ---------*/
        double* cn = nullptr;               // Coordination numbers                 [n]
        float* x = nullptr;                 // Positions, flattened                 [3n]
        double* f = nullptr;                // Forces, flattened                    [3n]
        double* dc6i = nullptr;             // dC6i(iat) saves dE_dsp/dCN(iat)      [n]
        /*--------- Per-atom values/arrays ---------*/

        /*--------- Per-pair values/arrays ---------*/
        float* c6_ij_tot = nullptr;
        float* dc6_iji_tot = nullptr;
        float* dc6_ijj_tot = nullptr;
        /*--------- Per-pair values/arrays ---------*/

        /*---------- Global values ---------*/
        int n_save;                         // to check whether the number of atoms has changed
        float disp_total;                   // Dispersion energy
        /*---------- Global values ---------*/

        /*--------- For loop over tau (translation of cell), flattened [range_x*range_y*range_z*3] ---------*/
        float* tau_vdw = nullptr;
        float* tau_cn = nullptr;
        int* tau_idx_vdw = nullptr;
        int* tau_idx_cn = nullptr;
        int tau_idx_vdw_total_size;
        int tau_idx_cn_total_size;
        int vdw_range_x_save = 0, vdw_range_y_save = 0, vdw_range_z_save = 0;
        int cn_range_x_save = 0, cn_range_y_save = 0, cn_range_z_save = 0;
        /*--------- For loop over tau (translation of cell) ---------*/

        /*--------- For device memory transfer (USM) ---------*/
        int *atomtype = nullptr;
        double *disp = nullptr;
        /*--------- For device memory transfer (USM) ---------*/
    };
}

#endif // LMP_PAIR_D3SYCL
#endif // PAIR_CLASS
