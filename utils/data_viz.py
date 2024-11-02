import matplotlib.pyplot as plt
import math

from patchify import unpatchify
from ipywidgets import interact, IntSlider, fixed
from data_loader.reconstruct_dataset import create_array_patches_per_image

import numpy as np

def visualize_reconstructed_images(images, masks, predictions, model_names, original_sizes, patches_size, patches_per_image, image_index):
    def update(z):
        num_models = len(predictions)
        
        # Calculate the number of rows needed for 3 columns
        num_plots = num_models + 2
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(18, 6 * num_plots))
        
        img_reshaped = images[image_index].reshape(patches_size[image_index])
        mask_reshaped = masks[image_index].reshape(patches_size[image_index])
        
        img = unpatchify(img_reshaped, original_sizes[image_index])[:, :, z]
        mask = unpatchify(mask_reshaped, original_sizes[image_index])[:, :, z]
        

        axes[0].imshow(img, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('Image')

        axes[1].imshow(mask, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('Ground Truth Mask')

        for i in range(num_models):
            predictions_and_its_masks = create_array_patches_per_image(predictions[i], patches_per_image)
            pred_reshaped = predictions_and_its_masks[image_index].reshape(patches_size[image_index])
            pred = unpatchify(pred_reshaped, original_sizes[image_index])[:, :, z]
            axes[i + 2].imshow(pred, cmap='gray')
            axes[i + 2].axis('off')
            axes[i + 2].set_title(model_names[i] + " Prediction")
            
        plt.tight_layout()
        plt.show()

    z_slider = IntSlider(
        value=0,
        min=0,
        max=original_sizes[image_index][2] - 1,
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

    img_2D = image[:, :, z]
    mask_2D = mask[:, :, z]

    axes[0, 0].imshow(img_2D, cmap='gray')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Image')

    axes[0, 1].imshow(mask_2D, cmap='gray')
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Ground Truth Mask')

    for i in range(num_models):
        row = (i + 2) // 3
        col = (i + 2) % 3
        pred_2D = predictions[i][patch_idx][:, :, z]
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