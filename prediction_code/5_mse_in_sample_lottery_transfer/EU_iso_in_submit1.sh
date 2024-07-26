#!/bin/bash
#SBATCH -A p31434               # Allocation
#SBATCH -p normal                # Queue
#SBATCH -t 48:00:00             # Walltime/duration of the job
#SBATCH -N 1                    # Number of Nodes
#SBATCH --mem=4G               # Memory per node in GB needed for a job. Also see --mem-per-cpu
#SBATCH --ntasks-per-node=1     # Number of Cores (Processors)
#SBATCH --mail-user=<my_email>  # Designate email address for job communications
##SBATCH --mail-type=<event>     # Events options are job BEGIN, END, NONE, FAIL, REQUEUE
#SBATCH --output=/projects/p31434/slurm_out/%A.%a.stdout    # Path for output must already exist
#SBATCH --error=/projects/p31434/slurm_out/%A.%a.stderr     # Path for errors must already exist
#SBATCH --job-name="5_iso1"       # Name of job
#SBATCH --array=0-4250
# 13243

source ~/.bashrc
conda activate ML_econ
cd "/projects/p31434/lottery/scripts/20240707/5_mse_in_sample_lottery_transfer"

time python EU_iso.py ${SLURM_ARRAY_TASK_ID}
