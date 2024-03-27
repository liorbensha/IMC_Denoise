import os
from os.path import join
from scipy import ndimage
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from tqdm import tqdm

input_dir = "IMC_Denoise_processed_image_folders"

# metrics to calculate
image_intensity_distributions = True # [True | False]
CNR_STDB = True # [True | False]
# SSIM = True # [True | False]
# PSNR = True # [True | False]

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


def img_intensities_distribution(fov_dir, protein, raw_img, pred_img, mask, input_dir):
    # Intersection of mask with raw and pred images
    raw_signal_pixels = raw_img * mask
    pred_signal_pixels = pred_img * mask
    # Intersection of opposite-mask with raw and pred images
    raw_bg_pixels = raw_img * (~mask)
    pred_bg_pixels = pred_img * (~mask)
    
    # Get non-zero pixel intensities
    raw_signal_intensities = raw_signal_pixels[np.nonzero(raw_signal_pixels)]
    pred_signal_intensities = pred_signal_pixels[np.nonzero(pred_signal_pixels)]
    raw_bg_intensities = raw_bg_pixels[np.nonzero(raw_bg_pixels)]
    pred_bg_intensities = pred_bg_pixels[np.nonzero(pred_bg_pixels)]
    
    # Calculate y-axis limits
    if raw_signal_intensities.sum() == 0:
        max_intensity = 10
    else:
        max_intensity = max(2.5 * np.percentile(raw_signal_intensities, 75) - 1.5 * np.percentile(raw_signal_intensities, 25), 
                            2.5 * np.percentile(pred_signal_intensities, 75) - 1.5 * np.percentile(pred_signal_intensities, 25))
        max_intensity *= 1.2  # Set the upper limit to 1.2 times the maximum whisker value
    min_intensity = 0
    
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle(f'FOV: {fov_dir}, Protein: {protein}')
    
    # Set style
    sns.set_theme(style="white")
    
    # Plot raw image box plot
    sns.boxplot(data=[raw_signal_intensities, raw_bg_intensities], ax=axs[0], showfliers=False, width=0.4, showmeans=True)
    axs[0].set_title('Raw Image')
    axs[0].set_ylabel('Intensity')
    axs[0].set_xticklabels(['Signal', 'Background'])
    axs[0].set_ylim([min_intensity, max_intensity])
    axs[0].text(0.5, 0.95, f'Non-zero signal pixels: {len(raw_signal_intensities)}', ha='center', transform=axs[0].transAxes, fontsize=8)
    axs[0].text(0.5, 0.9, f'Non-zero background pixels: {len(raw_bg_intensities)}', ha='center', transform=axs[0].transAxes, fontsize=8)
    
    # Plot pred image box plot
    sns.boxplot(data=[pred_signal_intensities, pred_bg_intensities], ax=axs[1], showfliers=False, width=0.4, showmeans=True)
    axs[1].set_title('Predicted Image')
    axs[1].set_ylabel('Intensity')
    axs[1].set_xticklabels(['Signal', 'Background'])
    axs[1].set_ylim([min_intensity, max_intensity])
    axs[1].text(0.5, 0.95, f'Non-zero signal pixels: {len(pred_signal_intensities)}', ha='center', transform=axs[1].transAxes, fontsize=8)
    axs[1].text(0.5, 0.9, f'Non-zero background pixels: {len(pred_bg_intensities)}', ha='center', transform=axs[1].transAxes, fontsize=8)
      
    plt.tight_layout()
    
    # Save figure according to protein
    protein_dir = os.path.join(input_dir, 'img_intensities_boxplots', 'according_to_protein', protein)
    os.makedirs(protein_dir, exist_ok=True)
    protein_path = os.path.join(protein_dir, f'{fov_dir}.png')
    plt.savefig(protein_path)
    
    # Save figure according to FOV
    fov_dir_path = os.path.join(input_dir, 'img_intensities_boxplots', 'according_to_FOV', fov_dir)
    os.makedirs(fov_dir_path, exist_ok=True)
    fov_path = os.path.join(fov_dir_path, f'{protein}.png')
    plt.savefig(fov_path)
    plt.close()


def CNR_STDB_metric(fov_dir, protein, target_image, signal_mask, CNR_df, STDB_df):
    # Get non-zero pixel values for signal and background regions
    signal_values = target_image[np.nonzero(signal_mask)]
    bg_values = target_image[np.nonzero(~signal_mask)]
    
    # Calculate CNR score
    CNR_score = (np.mean(signal_values) - np.mean(bg_values)) / np.std(bg_values)
    
    # Calculate STDB score
    STDB_score = np.std(bg_values)
    
    # Assign scores to CNR_df and STDB_df
    CNR_df.loc[fov_dir, protein] = CNR_score
    STDB_df.loc[fov_dir, protein] = STDB_score

def plot_protein_boxplots(input_dir, CNR_raw_df, CNR_pred_df, STDB_raw_df, STDB_pred_df):
    # Set style without grids
    sns.set_theme(style="white")
    # make output dir
    output_dir = join(input_dir, 'CNR_STDB')
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each protein
    for protein in CNR_raw_df.columns:
        # Create subplots for the current protein
        fig, axs = plt.subplots(1, 2, figsize=(8, 6))
        fig.suptitle(f'{protein}')

        # Plot CNR boxplot for raw and predicted images
        sns.boxplot(data=[CNR_raw_df[protein], CNR_pred_df[protein]], ax=axs[0], showfliers=False, showmeans=True, width=0.4)
        axs[0].set_title('CNR')
        axs[0].set_ylabel('CNR')
        axs[0].set_ylim(bottom=0)
        axs[0].set_xticklabels(['Raw', 'Predicted'])  # Set x-axis labels

        # Plot STDB boxplot for raw and predicted images
        sns.boxplot(data=[STDB_raw_df[protein], STDB_pred_df[protein]], ax=axs[1], showfliers=False, showmeans=True, width=0.4)
        axs[1].set_title('STDB')
        axs[1].set_ylabel('STDB')
        axs[1].set_ylim(bottom=0)
        axs[1].set_xticklabels(['Raw', 'Predicted'])  # Set x-axis labels

        plt.tight_layout()

        # Save figure for the current protein
        plt.savefig(join(output_dir, f'{protein}_boxplots.png'))
        plt.close()

# get lists of protein and FOVs
proteins_list = get_prtoein_list(input_dir)
fovs_list = [fov for fov in os.listdir(input_dir) if os.path.isdir(join(input_dir, fov, 'TIFs'))]

# create relevant dfs
if CNR_STDB:
    CNR_raw_df = pd.DataFrame(index=fovs_list, columns=proteins_list)
    CNR_pred_df = pd.DataFrame(index=fovs_list, columns=proteins_list)
    STDB_raw_df = pd.DataFrame(index=fovs_list, columns=proteins_list)
    STDB_pred_df = pd.DataFrame(index=fovs_list, columns=proteins_list)

# iterate over all FOVs and proteins
for fov_dir in tqdm(fovs_list, desc='FOV Directories'):
    fov_path = join(input_dir, fov_dir, 'TIFs')
    for protein in tqdm(proteins_list, desc=f'Proteins in {fov_dir}'):
        # get all relevant images
        clean_img = imread(join(fov_path, f'{protein}_clean.tif'))
        raw_img = imread(join(fov_path, f'{protein}_raw.tif'))
        pred_img = imread(join(fov_path, f'{protein}_pred.tif'))
        struct = ndimage.generate_binary_structure(2,2)
        mask = ndimage.binary_dilation(clean_img > 0, structure=struct, iterations=1)
        pred_image_pixelized = pred_img * (raw_img > 0)
        # merics
        if image_intensity_distributions:
            img_intensities_distribution(fov_dir, protein, raw_img, pred_img, mask, input_dir)
        if CNR_STDB:
            CNR_STDB_metric(fov_dir, protein, raw_img, mask, CNR_raw_df, STDB_raw_df)
            CNR_STDB_metric(fov_dir, protein, pred_img, mask, CNR_pred_df, STDB_pred_df)

if CNR_STDB:
    CNR_raw_df.to_csv(join(input_dir, 'CNR_raw_scores.csv'))
    CNR_pred_df.to_csv(join(input_dir, 'CNR_pred_scores.csv'))
    STDB_raw_df.to_csv(join(input_dir, 'STDB_raw_scores.csv'))
    STDB_pred_df.to_csv(join(input_dir, 'STDB_pred_scores.csv'))
    plot_protein_boxplots(input_dir, CNR_raw_df, CNR_pred_df, STDB_raw_df, STDB_pred_df)
