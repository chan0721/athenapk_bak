#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --account=jisuoqing_w
#SBATCH --nodes=1
#SBATCH --ntasks-per-nodes=1
#SBATCH --time=24:00:00


export PATH="/home/jisuoqing_w/project/ychen/athenapk_bak/cmake-4.2.1-linux-x86_64/bin:$PATH"
export PATH="/home/jisuoqing_w/project/ychen/gcc-13.2.0/bin:$PATH"
export PATH="/home/jisuoqing_w/project/sw/miniconda3/bin:$PATH"
export HDF5_ROOT=/home/jisuoqing_w/sw/hdf5-1.12.0/hdf5
export LD_LIBRARY_PATH=$HDF5_ROOT/lib:$LD_LIBRARY_PATH

module load nvidia/cuda/12.9


cd /home/jisuoqing_w/project/ychen/workcodes/turb_nocool_chi10/256

mpirun -np 8 /home/jisuoqing_w/project/ychen/athenapk/athenapk/build-gpu/bin/athenaPK -i /home/jisuoqing_w/project/ychen/athenapk_bak/athenapk/inputs/turb_nocool_chi10.in 