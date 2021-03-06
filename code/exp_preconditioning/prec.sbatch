#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=prec

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=03:30:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --array=43-51

#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=ludwig.bald@student.uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####

DATA_DIR=/scratch/$SLURM_JOB_ID/data_deepobs/

mkdir -p $DATA_DIR/pytorch
mkdir -p $DATA_DIR/pytorch/FashionMNIST/raw
cp -R /common/datasets/cifar10/. $DATA_DIR/pytorch/
cp -R /common/datasets/Fashion-MNIST/. $DATA_DIR/pytorch/FashionMNIST/raw/
####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
#cnn_minst.py <model save path> <mnist data path>
####



singularity exec ~/image.sif python3 ~/exp_preconditioning/runscript2.py --data_dir $DATA_DIR --random_seed $SLURM_ARRAY_TASK_ID


echo DONE!
