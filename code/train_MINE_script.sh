#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p cpu
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH -t 01-00:00:00
#SBATCH --qos=long-cpu
#SBATCH --job-name=MUS
source /etc/profile
source /home2/mxzq47/virtual_envs/music_venv/bin/activate
export TMPDIR=/dev/shm
python3 --version
python3 /home2/mxzq47/project/mxzq47_final/train.py --parameters MINE_parameters 

