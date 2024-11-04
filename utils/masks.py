from patchify import unpatchify
import numpy as np
from utils.data_viz import binarize_predictions
import os

from data_loader.reconstruct_dataset import create_matrix_images_as_rows_patches_as_cols

from tifffile import imsave

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_masks_inference(predictions, reshaped_patches_arr_sizes, nonreshaped_patches_arr_sizes, padded_image_sizes, nonpadded_image_sizes, dataset_name, model_names, images_names):
    
    create_dir('inference_masks')
    for model_idx in range(len(model_names)):
        predictions_patches_matrix = create_matrix_images_as_rows_patches_as_cols(predictions[model_idx], reshaped_patches_arr_sizes)
        for image_idx in range(len(images_names)):
            pred_reshaped = predictions_patches_matrix[image_idx].reshape(nonreshaped_patches_arr_sizes[image_idx])
            pred = unpatchify(pred_reshaped, padded_image_sizes[image_idx])[: nonpadded_image_sizes[image_idx][0], :nonpadded_image_sizes[image_idx][1], :nonpadded_image_sizes[image_idx][2]]
             
            imsave(f'inference_masks/{dataset_name}_{model_names[model_idx]}_{images_names[image_idx]}_mask.tif', binarize_predictions(pred))
    

    
        
        
        
        
            