#! /bin/bash

cd /mnt/nfs/scratch1/nnayak/peer-review-discourse-dataset/
conda deactivate
source ve/bin/activate
module load python3/3.9.0-2010

cd combined_alignment

python train.py -i torchtext_input_data_posneg_1_sample_1.0/ -t reg -r 2t
