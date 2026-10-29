#!/bin/bash

lammps_root=$1
cxx_standard=$2 # 14, 17
d3_support=$3 # 1, 0
flashTP_so="${4:-NONE}"
oeq_so="${5:-NONE}"
d3sycl_support="${6:-0}" # 1, 0 -- pair_d3sycl, pair_d3sycl_mpi (SYCL ports of pair_d3, pair_d3_mpi)
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
enable_flashTP=0
enable_oeq=0

###########################################
# Check if the given arguments are valid  #
###########################################

# Check the number of arguments
if [ "$#" -lt 3 ] || [ "$#" -gt 6 ]; then
    echo "Usage: sh patch_lammps.sh {lammps_root} {cxx_standard} {d3_support} {flashTP_so} {oeq_so} {d3sycl_support}"
    echo "  {lammps_root}: Root directory of LAMMPS source"
    echo "  {cxx_standard}: C++ standard (14, 17)"
    echo "  {d3_support}: Support for pair_d3 and pair_d3_mpi (d3, d3/mpi), CUDA (1, 0)"
    echo "  {d3sycl_support}: Support for pair_d3sycl and pair_d3sycl_mpi (d3sycl, d3sycl/mpi), SYCL (1, 0)."
    echo "                    Requires a SYCL compiler at LAMMPS configure time: AdaptiveCpp (discoverable via CMAKE_PREFIX_PATH) or Intel oneAPI icpx"
    echo "                    (CMAKE_CXX_COMPILER=icpx)."
    exit 1
fi

if [ "$d3_support" -ne 0 ] && [ "$d3sycl_support" -ne 0 ]; then
    echo "Error: d3_support (CUDA) and d3sycl_support (SYCL) cannot both be enabled"
    echo "       in the same LAMMPS build. add_sycl_to_target() wraps the whole 'lammps'"
    echo "       target's compile/link rules, including .cu files compiled by nvcc, which"
    echo "       conflicts with the CUDA language build used by pair_d3. Patch two separate"
    echo "       LAMMPS source trees instead, one per backend."
    exit 1
fi

# Check if the lammps_root directory exists
if [ ! -d "$lammps_root" ]; then
    echo "Error: No such directory: $lammps_root"
    exit 1
fi

# Check if the given directory is the root of LAMMPS source
if [ ! -d "$lammps_root/cmake" ] && [ ! -d "$lammps_root/potentials" ]; then
    echo "Error: Given $lammps_root is not a root of LAMMPS source"
    exit 1
fi

# pair_d3sycl's CMake block is inserted right after this line
if [ "$d3sycl_support" -ne 0 ] && ! grep -q '^add_library(lammps ${ALL_SOURCES})$' $lammps_root/cmake/CMakeLists.txt; then
    echo "Error: 'add_library(lammps \${ALL_SOURCES})' not found in $lammps_root/cmake/CMakeLists.txt"
    exit 1
fi

# Check if the script is being run from the root of SevenNet
if [ ! -f "${SCRIPT_DIR}/pair_e3gnn.cpp" ]; then
    echo "Error: Script executed in a wrong directory"
    exit 1
fi

if [ -f "$flashTP_so" ]; then
    echo "Using flashTP_so: $flashTP_so"
    enable_flashTP=1
elif [ "$flashTP_so" = "NONE" ]; then
    echo "[FlashTP] Skipped: not provided"
else
    echo "[FlashTP] Skipped: Invalid or missing flashTP_so given"
    flashTP_so="NONE"
fi

if [ -f "$oeq_so" ]; then
    echo "Using oeq_so: $oeq_so"
    enable_oeq=1
elif [ "$oeq_so" = "NONE" ]; then
    echo "[OEQ] Skipped: not provided"
else
    echo "[OEQ] Skipped: nvalid or missing oeq_so given"
    oeq_so="NONE"
fi

# Check if the patch is already applied
if [ -f "$lammps_root/src/pair_e3gnn.cpp" ]; then
    echo "----------------------------------------------------------"
    echo "Seems like given LAMMPS is already patched."
    echo "Try again after removing src/pair_e3gnn.cpp to force patch"
    echo "----------------------------------------------------------"
    echo "Example build commands, under LAMMPS root"
    echo "  mkdir build; cd build"
    echo "  cmake ../cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
    echo "  make -j 4"
    exit 0
fi

# Check if OpenMPI exists and if it is CUDA-aware
if command -v ompi_info &> /dev/null; then
    cuda_support=$(ompi_info --parsable --all | grep mpi_built_with_cuda_support:value)
    if [[ -z "$cuda_support" ]]; then
        echo "OpenMPI not found, parallel performance is not optimal"
    elif [[ "$cuda_support" == *"true" ]]; then
        echo "OpenMPI is CUDA aware"
    else
        echo "This system's OpenMPI is not 'CUDA aware', parallel performance is not optimal"
    fi
else
    echo "OpenMPI not found, parallel performance is not optimal"
fi

# Extract LAMMPS version and update
lammps_version=$(grep "#define LAMMPS_VERSION" $lammps_root/src/version.h | awk '{print $3, $4, $5}' | tr -d '"')

# Combine version and update
detected_version="$lammps_version"
required_version="2 Aug 2023"  # Example required version

# Check if the detected version is compatible
if [[ "$detected_version" != "$required_version" ]]; then
    echo "Warning: Detected LAMMPS version ($detected_version) may not be compatible. Required version: $required_version"
fi

###########################################
# Backup original LAMMPS source code      #
###########################################

# Create a backup directory if it doesn't exist
backup_dir="$lammps_root/_backups"
mkdir -p $backup_dir

# Copy comm_* from original LAMMPS source as backup
cp $lammps_root/src/comm_brick.cpp $backup_dir/
cp $lammps_root/src/comm_brick.h $backup_dir/

# Copy cmake/CMakeLists.txt from original source as backup
cp $lammps_root/cmake/CMakeLists.txt $backup_dir/CMakeLists.txt

###########################################
# Patch LAMMPS source code: e3gnn         #
###########################################

# 1. Copy pair_e3gnn files to LAMMPS source
cp $SCRIPT_DIR/{pair_e3gnn,pair_e3gnn_parallel,comm_brick}.cpp $lammps_root/src/
cp $SCRIPT_DIR/{pair_e3gnn,pair_e3gnn_parallel,comm_brick}.h $lammps_root/src/
# Always copy the oEq autograd bridge (pair_e3gnn.cpp has an extern reference to it)
cp $SCRIPT_DIR/pair_e3gnn_oeq_autograd.cpp $lammps_root/src/  # TODO: set this as oeq-specific

# 2. Patch cmake/CMakeLists.txt
sed -i "s/set(CMAKE_CXX_STANDARD 11)/set(CMAKE_CXX_STANDARD $cxx_standard)/" $lammps_root/cmake/CMakeLists.txt
cat >> $lammps_root/cmake/CMakeLists.txt << "EOF"

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
EOF

optional_libs=()
optional_rpaths=()

if [ "$enable_flashTP" -eq 1 ]; then
    optional_libs+=("    \${CMAKE_CURRENT_LIST_DIR}/flashTP/libflashtp_large_kernel_lammps.so")
    optional_rpaths+=("\${CMAKE_CURRENT_LIST_DIR}/flashTP")
    mkdir -p $lammps_root/cmake/flashTP && cp $flashTP_so $lammps_root/cmake/flashTP/libflashtp_large_kernel_lammps.so && echo "[FlashTP] flashTP so file copied"
fi

if [ "$enable_oeq" -eq 1 ]; then
    optional_libs+=("    \${CMAKE_CURRENT_LIST_DIR}/oeq/liboeq_stable_cuda_aoti.so")
    optional_rpaths+=("\${CMAKE_CURRENT_LIST_DIR}/oeq")
    mkdir -p $lammps_root/cmake/oeq && cp $oeq_so $lammps_root/cmake/oeq/liboeq_stable_cuda_aoti.so && echo "[OEQ] oeq so file copied"
fi

if [ ${#optional_libs[@]} -gt 0 ]; then
    rpath_joined=$(IFS=';'; echo "${optional_rpaths[*]}")
    {
        echo ""
        echo "find_package(Python3 REQUIRED COMPONENTS Development)"
        echo "target_link_libraries(lammps PUBLIC -Wl,--no-as-needed"
        for lib in "${optional_libs[@]}"; do
            echo "$lib"
        done
        echo "    Python3::Python"
        echo ")"
        echo "set_property(TARGET lammps APPEND PROPERTY"
        echo "    BUILD_RPATH \"$rpath_joined\""
        echo ")"
    } >> $lammps_root/cmake/CMakeLists.txt
    echo "[Optional] CMakeLists.txt is patched"
fi


###########################################
# Patch LAMMPS source code: d3            #
###########################################

if [ "$d3_support" -ne 0 ]; then

# 1. Copy pair_d3 files to LAMMPS source
cp $SCRIPT_DIR/{pair_d3,pair_d3_mpi}.cu $lammps_root/src/
cp $SCRIPT_DIR/{pair_d3,pair_d3_mpi}.h $lammps_root/src/
cp $SCRIPT_DIR/pair_d3_pars.h $lammps_root/src/

# 2. Patch cmake/CMakeLists.txt
sed -i "s/project(lammps CXX)/project(lammps CXX CUDA)/" $lammps_root/cmake/CMakeLists.txt
sed -i "s/\${LAMMPS_SOURCE_DIR}\/\[\^.\]\*\.cpp/\${LAMMPS_SOURCE_DIR}\/\[\^.\]\*\.cpp  \${LAMMPS_SOURCE_DIR}\/\[\^.\]\*\.cu/" $lammps_root/cmake/CMakeLists.txt
cat >> $lammps_root/cmake/CMakeLists.txt << "EOF"

find_package(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fmad=false -O3")
string(REPLACE "-gencode arch=compute_50,code=sm_50" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
target_link_libraries(lammps PUBLIC ${CUDA_LIBRARIES} cuda)
EOF

fi

###########################################
# Patch LAMMPS source code: d3sycl        #
###########################################

if [ "$d3sycl_support" -ne 0 ]; then

# 1. Copy pair_d3sycl files to LAMMPS source
cp $SCRIPT_DIR/{pair_d3sycl,pair_d3sycl_mpi}.h $lammps_root/src/
cp $SCRIPT_DIR/{pair_d3sycl,pair_d3sycl_mpi}.cpp $lammps_root/src/
if [ ! -f "$lammps_root/src/pair_d3_pars.h" ]; then
    cp $SCRIPT_DIR/pair_d3_pars.h $lammps_root/src/
fi

# 2. Patch cmake/CMakeLists.txt: link the "lammps" target against SYCL.
# AdaptiveCpp must be discoverable, e.g. via
#   -DCMAKE_PREFIX_PATH=/path/to/adaptivecpp/install
sed -i "/^add_library(lammps \${ALL_SOURCES})\$/a\\
\\
# --- SYCL (AdaptiveCpp) support for pair_d3sycl ---\\
#find_package(AdaptiveCpp REQUIRED)\\
\\
# Find Intel oneAPI Base Toolkit\\
if(NOT DEFINED ONEAPI_ROOT)\\
    if(DEFINED ENV{ONEAPI_ROOT})\\
        set(ONEAPI_ROOT $ENV{ONEAPI_ROOT})\\
    elseif(WIN32)\\
        set(ONEAPI_ROOT "C:/Program Files (x86)/Intel/oneAPI")\\
    else()\\
        set(ONEAPI_ROOT "/opt/intel/oneapi")\\
    endif()\\
endif()\\
\\
# Source Intel oneAPI environment if available\\
if(WIN32 AND EXISTS "${ONEAPI_ROOT}/setvars.bat")\\
    execute_process(\\
        COMMAND cmd /d /c "call \"${ONEAPI_ROOT}/setvars.bat\" >nul 2>&1 && set"\\
        OUTPUT_VARIABLE ONEAPI_ENV\\
    )\\
elseif(EXISTS "${ONEAPI_ROOT}/setvars.sh")\\
    execute_process(\\
        COMMAND bash -c "source ${ONEAPI_ROOT}/setvars.sh > /dev/null 2>&1 && env"\\
        OUTPUT_VARIABLE ONEAPI_ENV\\
    )\\
endif()\\
\\
if(DEFINED ONEAPI_ENV)\\
    string(REPLACE "\r\n" "\n" ONEAPI_ENV "${ONEAPI_ENV}")\\
    string(REGEX MATCHALL "([^=\n]+)=([^\n]*)" ONEAPI_ENV_MATCHES "${ONEAPI_ENV}")\\
    foreach(MATCH ${ONEAPI_ENV_MATCHES})\\
        string(REGEX REPLACE "([^=\n]+)=([^\n]*)" "\\1" KEY "${MATCH}")\\
        string(REGEX REPLACE "([^=\n]+)=([^\n]*)" "\\2" VALUE "${MATCH}")\\
        set(ENV{${KEY}} ${VALUE})\\
    endforeach()\\
endif()\\
\\
# Check for Intel SYCL compiler\\
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")\\
    set(SYCL_COMPILER_FOUND TRUE)\\
else()\\
    if(WIN32)\\
        find_program(SYCL_COMPILER NAMES icx icx-cl)\\
    else()\\
        find_program(SYCL_COMPILER icpx)\\
    endif()\\
    if(SYCL_COMPILER)\\
        set(SYCL_COMPILER_FOUND TRUE)\\
        set(CMAKE_CXX_COMPILER ${SYCL_COMPILER})\\
    endif()\\
endif()\\
\\
# SYCL_COMPILER_FOUND \\
message(STATUS "Found SYCL compiler: ${CMAKE_CXX_COMPILER}")\\
\\
# Add SYCL include directories\\
find_path(SYCL_INCLUDE_DIR\\
    NAMES \\
        "CL/sycl.hpp"\\
        "sycl/sycl.hpp"\\
    PATHS\\
    "${ONEAPI_ROOT}/compiler/latest/linux/include"\\
    "${ONEAPI_ROOT}/compiler/latest/include"\\
    "/opt/intel/oneapi/compiler/latest/linux/include"\\
    "/opt/intel/oneapi/compiler/latest/include"\\
    "/usr/include"\\
    "/usr/local/include"\\
    ENV CPLUS_INCLUDE_PATH\\
)\\
\\
message(STATUS "Found SYCL include directory: ${SYCL_INCLUDE_DIR}")\\
include_directories(${SYCL_INCLUDE_DIR} ${SYCL_INCLUDE_DIR}/..)\\
\\
# Set SYCL-specific compiler flags\\
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")\\
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsycl")\\
\\
\\
#add_sycl_to_target(TARGET lammps SOURCES \${LAMMPS_SOURCE_DIR}/pair_d3sycl.cpp)\\
add_sycl_to_target(TARGET lammps SOURCES \${LAMMPS_SOURCE_DIR}/pair_d3sycl.cpp\\
                                           \${LAMMPS_SOURCE_DIR}/pair_d3sycl_mpi.cpp)\\
# --- SYCL (AdaptiveCpp) support for pair_d3sycl ---" $lammps_root/cmake/CMakeLists.txt
fi

###########################################
# Print changes and backup file locations #
###########################################

# Print changes and backup file locations
echo "Changes made:"
echo "  - Original LAMMPS files (src/comm_brick.*, cmake/CMakeList.txt) are in {lammps_root}/_backups"
echo "  - Copied contents of pair_e3gnn to $lammps_root/src/"
echo "  - Patched CMakeLists.txt: include LibTorch, CXX_STANDARD $cxx_standard"
echo
if [ "$enable_flashTP" -eq 1 ]; then
    echo "  - Copied flashTP .so to $lammps_root/cmake/flashTP/"
    echo "  - Patched CMakeLists.txt: link libflashtp_large_kernel_lammps.so"
    echo
fi
if [ "$enable_oeq" -eq 1 ]; then
    echo "  - Copied oeq .so to $lammps_root/cmake/oeq/"
    echo "  - Copied pair_e3gnn_oeq_autograd.cpp to $lammps_root/src/"
    echo "  - Patched CMakeLists.txt: link liboeq_stable_cuda_aoti.so"
    echo
fi
if [ "$d3_support" -ne 0 ]; then
    echo "  - Copied pair_d3.{h,cu}, pair_d3_mpi.{h,cu} and pair_d3_pars.h to $lammps_root/src/"
    echo "  - pair_style names: d3 (single MPI rank only), d3/mpi (multi-rank / multi-GPU)"
    echo "  - Patched CMakeLists.txt: include CUDA"
    echo
fi
if [ "$d3sycl_support" -ne 0 ]; then
    echo "  - Copied pair_d3sycl.{h,cpp}, pair_d3sycl_mpi.{h,cpp} (and pair_d3_pars.h, if missing) to $lammps_root/src/"
    echo "  - Patched CMakeLists.txt: SYCL build of the lammps target (AdaptiveCpp or Intel icpx)"
    echo "  - Build requires a SYCL compiler at configure time, e.g.:"
    echo "      AdaptiveCpp: cmake -DCMAKE_PREFIX_PATH=/path/to/adaptivecpp/install -DACPP_TARGETS=generic ..."
    echo "      Intel oneAPI: source /opt/intel/oneapi/setvars.sh; cmake -DCMAKE_CXX_COMPILER=icpx ..."
    echo "  - pair_style names: d3sycl (single MPI rank only), d3sycl/mpi (multi-rank / multi-GPU); same arguments as d3"
    echo
fi
# Provide example cmake command to the user
echo "Example build commands, under LAMMPS root"
echo "  mkdir build; cd build"
echo "  cmake ../cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
echo "  make -j 4"

exit 0
