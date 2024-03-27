#!/bin/bash

get_protein_list() {
python3 <<END
import os

def get_protein_list(input_dir):
    protein_names = set()
    for fov_dir in os.listdir(input_dir):
        fov_path = os.path.join(input_dir, fov_dir, 'TIFs')
        if os.path.exists(fov_path) and os.path.isdir(fov_path):
            for protein_image in os.listdir(fov_path):
                protein_path = os.path.join(fov_path, protein_image)
                if os.path.isfile(protein_path) and not protein_image.startswith('.'):
                    protein_name, _ = os.path.splitext(protein_image)
                    protein_names.add(protein_name)
    return list(protein_names)

print(' '.join(get_protein_list('../Data/Test/noisy/Ofer_PDAC')))
END
}
protein_names=($(get_protein_list))

# # Print the protein_names variable
# echo "Protein names:"
# for name in "${protein_names[@]}"; do
#     echo "$name"
# done

# Loop through each protein name
for protein_name in "${protein_names[@]}"
do
    # Execute the job using bsub
    bsub <<EOF
#BSUB -J IMC_Denoise_inference_${protein_name}
#BSUB -q leeat-gpu
#BSUB -R rusage[mem=100GB]
#BSUB -R affinity[thread*2]
#BSUB -gpu num=1:j_exclusive=yes:gmem=40GB

# activate conda env
source activate IMC_Denoise;  

# run this file

# DIMR+DeepSNiF
python3 scripts/Predict_IMC_Denoise_batch.py \
    --channel_name '${protein_name}' \
    --load_directory '../Data/Test/noisy/Ofer_PDAC' \
    --save_directory 'results/Ofer_Test' \
    --weights_name 'weights_${protein_name}.hdf5' \
    --weights_save_directory 'trained_weights' \
    --n_neighbours '4' \
    --n_iter '3' \
    --slide_window_size '3' \
    --network_size 'normal'
EOF
done

