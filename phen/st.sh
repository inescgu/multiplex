#!/bin/bash
#SBATCH --job-name=sto
#SBATCH --output=slurm_logs/sto_%j.out
#SBATCH --error=slurm_logs/sto_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1     

cd /lisc/home/user/gerardursin/og_gene_set/phen

# Create necessary directories
mkdir -p slurm_logs

# Define jobs - all the same script but with different arguments
declare -a job_scripts=(
 "analyze_spatial_transcriptomics.py" # seed_lcc endo reactome

)

# Define arguments for each job
declare -a job_args=(
   "--ppi_source combined26 --input_dir ../output/combined26 --output_dir ./output --expression_data ../../expression_data/spatial_transcriptomics_drg.csv"
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
pip install statsmodels

# Run the job with arguments
python $current_job $current_args

# End time
echo "Finished at: $(date)"