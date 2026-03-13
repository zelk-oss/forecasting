#!/bin/bash
name="test_run"
outdir="outputs"

CKPT="./ckpts/latest.pt"
DATA="../dataset/test_128_dt10.pt"
N_SAMPLES=16
N_ENSEMBLE=6
N_LAG=5
N_STEPS=200

echo "Launching test job: $name"
echo "  checkpoint : $CKPT"
echo "  n_samples  : $N_SAMPLES  |  n_lag: $N_LAG  |  n_steps: $N_STEPS"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH -A wbg@v100
#SBATCH -C v100-32g
#SBATCH --partition=gpu_p13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4               # testing is lighter than training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=${outdir}/%x_%j.log
#SBATCH --error=${outdir}/%x_%j.log
#SBATCH --time=01:00:00                 # 1h is plenty for inference
#SBATCH --hint=nomultithread

module purge
module load pytorch-gpu/py3/2.3.0

export WANDB_MODE=offline

set -x
mkdir -p ${outdir}

srun python test.py \
    --ckpt       ${CKPT}       \
    --data       ${DATA}       \
    --n_samples  ${N_SAMPLES}  \
    --n_ensemble ${N_ENSEMBLE} \
    --n_lag      ${N_LAG}      \
    --n_steps    ${N_STEPS}    \
    --device     cuda:0
EOT
