import os
from os.path import join
from scipy import ndimage
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
import seaborn as sns
from sklearn.metrics import f1_score
from tqdm import tqdm

input_dir = "IMC_Denoise_processed_image_folders"

# get proteins list from the given directory
def get_prtoein_list(input_dir):
    protein_names = set()  # Using a set to ensure unique protein names
    for fov_dir in os.listdir(input_dir):
        fov_path = os.path.join(input_dir, fov_dir, 'TIFs')
        if os.path.exists(fov_path) and os.path.isdir(fov_path):
            # Inside 'TIFs', iterate over each protein image
            for protein_image in os.listdir(fov_path):
                protein_path = os.path.join(fov_path, protein_image)
                
                if os.path.isfile(protein_path) and not protein_image.startswith('.'):
                    # Extract protein name without suffix
                    protein_name, _ = os.path.splitext(protein_image)          
                    # Determine suffix of the protein
                    if protein_name.endswith('_raw'):
                        protein_names.add(protein_name[:-4])
    # convert to list
    return list(protein_names)


def find_best_F1_threshold(denoised_img, clean_img):
    precision, recall, thresholds = precision_recall_curve(clean_img.ravel() > 0, (denoised_img / 255.0).ravel())
    # plt.figure(figsize=(8, 6))
    # plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    # plt.plot([0, 1], [0.5, .5], color='gray', linestyle='--')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.legend(loc='lower left')
    # plt.grid(True)
    # plt.savefig('precision_recall_curve.png')

    # calculate F1 scores
    f1_scores = 2 * (precision * recall) / (precision + recall + 10 ** -10)

    # Find the threshold with the highest F1 score
    best_threshold = thresholds[np.argmax(f1_scores)] * 255
    best_f1_score = max(f1_scores)
    
    # Apply the best threshold to denoised image
    thresholded_denoised_img = denoised_img * (denoised_img >= best_threshold)
    return thresholded_denoised_img, '{:.3f}'.format(best_f1_score), '{:.3f}'.format(best_threshold)


# get lists of protein and FOVs
proteins_list = get_prtoein_list(input_dir)
fovs_list = [fov for fov in os.listdir(input_dir) if os.path.isdir(join(input_dir, fov, 'TIFs'))]

# create dfs
F1_pred_df = pd.DataFrame(index=fovs_list, columns=proteins_list)
F1_raw_df = pd.DataFrame(index=fovs_list, columns=proteins_list)
th_pred_df = pd.DataFrame(index=fovs_list, columns=proteins_list)
th_raw_df = pd.DataFrame(index=fovs_list, columns=proteins_list)
# iterate over all FOVs and proteins
# for fov_dir in tqdm(fovs_list, desc='FOV Directories'):
for fov_dir in fovs_list:
    fov_path = join(input_dir, fov_dir, 'TIFs')
    # for protein in tqdm(proteins_list, desc=f'Proteins in {fov_dir}'):
    for protein in proteins_list:
        # get all relevant images
        clean_img = imread(join(fov_path, f'{protein}_clean.tif'))
        raw_img = imread(join(fov_path, f'{protein}_raw.tif'))
        pred_img = imread(join(fov_path, f'{protein}_pred.tif'))
        struct = ndimage.generate_binary_structure(2,2)
        mask = ndimage.binary_dilation(clean_img > 0, structure=struct, iterations=2)
        pred_image_pixelized = pred_img * (raw_img > 0)
        thresholded_pred_img, F1_pred_df.loc[fov_dir, protein], th_pred_df.loc[fov_dir, protein] = find_best_F1_threshold(pred_image_pixelized, clean_img)
        imsave(join(fov_path, f'{protein}_pred_th_{F1_pred_df.loc[fov_dir, protein]}.tif'), thresholded_pred_img, check_contrast=False)
        thresholded_raw_img, F1_raw_df.loc[fov_dir, protein], th_raw_df.loc[fov_dir, protein] = find_best_F1_threshold(raw_img, clean_img)
        print(f'F1_pred: {F1_pred_df.loc[fov_dir, protein]}, F1_raw: {F1_raw_df.loc[fov_dir, protein]}')
        imsave(join(fov_path, f'{protein}_raw_th_{F1_raw_df.loc[fov_dir, protein]:.3f}.tif'), thresholded_raw_img, check_contrast=False)

F1_pred_df.to_csv(join(input_dir, 'F1_pred_df.csv'))
F1_raw_df.to_csv(join(input_dir, 'F1_raw_df.csv'))
th_pred_df.to_csv(join(input_dir, 'th_pred_df.csv'))
th_raw_df.to_csv(join(input_dir, 'th_raw_df.csv'))

        