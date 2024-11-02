from tensorflow.keras.models import load_model
from data_loader.reconstruct_dataset import create_dataset_inference, create_array_patches_per_image
from utils.data_viz import visualize_patches_3D_in_2D, visualize_reconstructed_images
from ipywidgets import IntSlider, interact, fixed
from utils.metrics import plot_violin

import matplotlib.pyplot as plt

from patchify import unpatchify

import numpy as np

def predict_model(test_dataset, model_path):
    model  = load_model(model_path)
    
    predictions = model.predict(test_dataset, batch_size=16)
    
    return predictions

def inference(
    test_dir, 
    model_paths, 
    model_names,
    violin_plot_filename,
    patch_shape,
    patch_step,
): 
    dataset, reconstruction_info = create_dataset_inference(
        test_dir,
        patch_shape=patch_shape, 
        patch_step=patch_step,
    )
    
    predictions = []
    
    for model_path in model_paths: 
        predictions.append(predict_model(dataset[0], model_path))
    
    patch_slider = IntSlider(
        value=0,
        min=0,
        max=dataset[0].shape[0] - 1,
        step=1,
        description='Current Patch: ',
        continuous_update=False
    )
    
    z_slider = IntSlider(
        value=0,
        min=0,
        max=patch_shape[2] - 1,
        step=1,
        description='Z value:',
        continuous_update=False
    )
    
    interact(visualize_patches_3D_in_2D, dataset=fixed(dataset), predictions=fixed(predictions), model_names=fixed(model_names), patch_idx=patch_slider, z=z_slider)

    
    images_and_its_patches = create_array_patches_per_image(dataset[0], reconstruction_info[2])
    masks_and_its_patches = create_array_patches_per_image(dataset[1], reconstruction_info[2])
        
    image_slider = IntSlider(
        value=0,
        min=0,
        max= len(images_and_its_patches) - 1,
        step=1,
        description='Current Image: ',
        continuous_update=False
    )
    
    interact(visualize_reconstructed_images, images=fixed(images_and_its_patches), masks=fixed(masks_and_its_patches), predictions=fixed(predictions), model_names=fixed(model_names), original_sizes=fixed(reconstruction_info[0]), patches_size=fixed(reconstruction_info[1]), patches_per_image=fixed(reconstruction_info[2]), image_index=image_slider)
    
    plot_violin(predictions, dataset[1], model_names, violin_plot_filename)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    