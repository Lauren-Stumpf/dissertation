#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p cpu
#SBATCH -t 00-00:10:00
#SBATCH --qos=long-cpu
#SBATCH --job-name=MUS
source /etc/profile
source /home2/mxzq47/virtual_envs/music_venv/bin/activate
export TMPDIR=/dev/shm
python3 --version
python3 /home2/mxzq47/project/new_train/save_weight.py --location /home2/mxzq47/project/new_train/logs/FetchPush-v1-T-2022-05-02-14-48-52/policy_best.pkl


