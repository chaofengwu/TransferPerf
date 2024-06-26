#!/bin/bash
#SBATCH -A p31434               # Allocation
#SBATCH -p short                # Queue
#SBATCH -t 4:00:00             # Walltime/duration of the job
#SBATCH -N 1                    # Number of Nodes
#SBATCH --mem=4G               # Memory per node in GB needed for a job. Also see --mem-per-cpu
#SBATCH --ntasks-per-node=1     # Number of Cores (Processors)
#SBATCH --mail-user=<my_email>  # Designate email address for job communications
##SBATCH --mail-type=<event>     # Events options are job BEGIN, END, NONE, FAIL, REQUEUE
#SBATCH --output=/projects/p31434/slurm_out/%j.%N.stdout    # Path for output must already exist
#SBATCH --error=/projects/p31434/slurm_out/%j.%N.stderr     # Path for errors must already exist
#SBATCH --job-name="cv iso"       # Name of job
#SBATCH --array=0-439
#2199

source ~/.bashrc
conda activate ML_econ
cd "/projects/p31434/lottery/scripts/20220606/cv_mse"

time python EU_iso.py ${SLURM_ARRAY_TASK_ID}
