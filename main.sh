#!/bin/bash
#SBATCH -p sched_mit_kburdge_r8      # partition name
#SBATCH --job-name=myjob             # name for your job
#SBATCH --gres=gpu:1                 # if you need GPUs
#SBATCH --ntasks=1                   # number of tasks (often 1 for serial jobs)
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem=16G                    # memory per node
#SBATCH --time=02:00:00              # max walltime (HH:MM:SS)
#SBATCH --output=slurm-%j.out        # output file (%j = job ID) to capture logs for debugging

# # Load your shell environment to activate your Conda environment
# source /home/user/.bashrc
# conda activate myconda
VENV_PATH="/orcd/home/002/josiexw/ondemand/data/sys/myjobs/projects/els-audio/venv" 

# Check if the virtual environment exists
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment at $VENV_PATH..."
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Load any modules or software you need
module load cuda/12.0

# Run your command or script
python my_analysis.py