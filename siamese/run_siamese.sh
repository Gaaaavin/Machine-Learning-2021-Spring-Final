#!/bin/bash
#SBATCH --job-name=Siamese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --gres=gpu:aquila
#SBATCH --time=50:00:00
#SBATCH --output=siamese_%j.out
#SBATCH --gres=gpu:1 # How much gpu need, n is the number
#SBATCH -p aquila


module purge
module load anaconda3
module load cuda/10.0
module load gcc/7.3
cd /gpfsnyu/home/xl3136/siamese

echo "start training"
source activate ml
python3 siamese.py
echo "FINISH"