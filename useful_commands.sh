module avail
module load anaconda3/2025.6

salloc --nodes=1 --ntasks=1 --mem=64G --time=01:01:00 --gres=gpu:4 --partition=pli-c --mail-type=begin

squeue -u xw2259

cd /scratch/gpfs/xw2259

shownodes -p pli-c

sbatch job.sh