#!/bin/bash
name="training_run"
outdir="outputs"

BETA='t^2'
SIGMA=0.1

echo "Launching training job: $name"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH -A wbg@v100                      # account 
#SBATCH -C v100-32g                      # V100 with 32GB memory
#SBATCH --partition=gpu_p13              # default V100 partition
#SBATCH --gres=gpu:1                     # 1 GPU
#SBATCH --cpus-per-task=10               # gives you ~40GB RAM (standard for 1 GPU on this partition)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/%x_%j.log       # %x=job name, %j=job ID
#SBATCH --error=outputs/%x_%j.log
#SBATCH --time=12:00:00                  # slightly above 11h as buffer
#SBATCH --hint=nomultithread
# NO --qos line = uses default qos_gpu-t3, which allows up to 20 hours

module purge
module load pytorch-gpu/py3/2.3.0

set -x
mkdir -p $outdir

srun python main.py --beta_fn ${BETA} --sigma_coef ${SIGMA} --use_wandb 1 --debug 0
EOT