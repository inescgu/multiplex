#!/bin/bash
#SBATCH --job-name=proto
#SBATCH --output=slurm_logs/proto_%j.out
#SBATCH --error=slurm_logs/proto_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1     

cd /lisc/home/user/gerardursin/og_gene_set/phen



# Define jobs - all the same script but with different arguments
declare -a job_scripts=(
 "analyze_proteomics.py" 
)

# Define arguments for each job
declare -a job_args=(
"--ppi_source combined26 --input_dir ../output/combined26 --expression_data ../../expression_data/matrix-for-limma2.csv --metadata_file ../../expression_data/colData-for-limma.csv
"
)

# Get the current job from the array
current_job=${job_scripts[$SLURM_ARRAY_TASK_ID-1]}
current_args=${job_args[$SLURM_ARRAY_TASK_ID-1]}

# Print some information about the job
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Running job: $current_job"
echo "With arguments: $current_args"

# Load any required modules here
module load Python/3.14.4-weekly  
source /lisc/data/scratch/menche/ines/general_env/bin/activate
# Run the job with arguments
python $current_job $current_args

# End time
echo "Finished at: $(date)"