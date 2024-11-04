import matplotlib.pyplot as plt
import math

from patchify import unpatchify
from ipywidgets import interact, IntSlider, fixed
from data_loader.reconstruct_dataset import create_matrix_images_as_rows_patches_as_cols

import numpy as np

def visualize_reconstructed_images(images, masks, predictions, model_names, nonreshaped_patches_arr_sizes, reshaped_patches_arr_sizes ,nonpadded_image_sizes, padded_image_sizes, dataset_name, image_index):
    def update(z):
        num_models = len(predictions)
        
        # Calculate the number of rows needed for 3 columns
        num_plots = num_models + 2
        num_rows = math.ceil(num_plots / 3)
        
        fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_plots))
        
        img_reshaped = images[image_index].reshape(nonreshaped_patches_arr_sizes[image_index])
        mask_reshaped = masks[image_index].reshape(nonreshaped_patches_arr_sizes[image_index])
        
        img = unpatchify(img_reshaped, padded_image_sizes[image_index])[:nonpadded_image_sizes[image_index][0], :nonpadded_image_sizes[image_index][1], :nonpadded_image_sizes[image_index][2]]
        mask = unpatchify(mask_reshaped, padded_image_sizes[image_index])[:nonpadded_image_sizes[image_index][0], :nonpadded_image_sizes[image_index][1], :nonpadded_image_sizes[image_index][2]]
        
        axes[0, 0].imshow(binarize_predictions(img[z, :, :]), cmap='gray')
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Image')

        axes[0, 1].imshow(binarize_predictions(mask[z, :, :]), cmap='gray')
        axes[0, 1].axis('off')
        axes[0, 1].set_title('Ground Truth Mask')

        for i in range(num_models):
            row = (i + 2) // 3
            col = (i + 2) % 3
            prediction_patches_matrix = create_matrix_images_as_rows_patches_as_cols(predictions[i], reshaped_patches_arr_sizes)
            pred_reshaped = prediction_patches_matrix[image_index].reshape(nonreshaped_patches_arr_sizes[image_index])
            pred = unpatchify(pred_reshaped, padded_image_sizes[image_index])[: nonpadded_image_sizes[image_index][0], :nonpadded_image_sizes[image_index][1], :nonpadded_image_sizes[image_index][2]]
                        
            axes[row, col].imshow(binarize_predictions(pred[z, :, :]), cmap='gray')
            axes[row, col].axis('off')
            axes[row, col].set_title(model_names[i] + " Prediction")
        
        # Hide any unused subplots
        for ax in axes[num_plots:]:
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()

    z_slider = IntSlider(
        value=0,
        min=0,
        max=nonpadded_image_sizes[image_index][0] - 1,
        step=1,
        description='Z value:',
        continuous_update=False
    )

    interact(update, z=z_slider)
    

def visualize_patches_3D_in_2D(dataset, predictions, model_names, patch_idx, z):
    image = dataset[0][patch_idx]
    mask = dataset[1][patch_idx]
    num_models = len(predictions)
    
    # Calculate the number of rows needed for 3 columns
    num_plots = num_models + 2
    num_rows = math.ceil(num_plots / 3)
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows))

    img_2D = image[z, :, :]
    mask_2D = mask[z, :, :]

    axes[0, 0].imshow(img_2D, cmap='gray')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Image')

    axes[0, 1].imshow(mask_2D, cmap='gray')
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Ground Truth Mask')

    for i in range(num_models):
        row = (i + 2) // 3
        col = (i + 2) % 3
        pred_2D = binarize_predictions(predictions[i][patch_idx][z, :, :])
        axes[row, col].imshow(pred_2D, cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(model_names[i] + " Prediction")

    # Hide any unused subplots
    for i in range(num_plots, num_rows * 3):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()
    
def binarize_predictions(predictions, threshold=0.5):
    return (predictions >= threshold).astype(int)
