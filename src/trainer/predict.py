from tensorflow.keras.models import load_model
from data_loader.reconstruct_dataset import create_dataset_inference, create_dataset_prediction, create_matrix_images_as_rows_patches_as_cols

from utils.data_viz import visualize_patches_3D_in_2D, visualize_reconstructed_images, binarize_predictions
from utils.masks import save_masks_inference
from ipywidgets import IntSlider, interact, fixed
from utils.metrics import plot_violin

import matplotlib.pyplot as plt

from patchify import unpatchify

import numpy as np

def predict_model(test_dataset, model_path, batch_size):
    model  = load_model(model_path)
    predictions = model.predict(test_dataset, batch_size)
    
    return predictions

def inference(
    input_dir, 
    model_path, 
    model_name,
    patch_shape,
    patch_step,
    batch_size,
    out_dir
): 
    all_images_names, datasets, reconstruction_info = create_dataset_prediction(
        input_dir,
        patch_shape=patch_shape, 
        patch_step=patch_step,
    )

    predictions = binarize_predictions(predict_model(datasets, model_path, batch_size))

    save_masks_inference(predictions, reconstruction_info[3], reconstruction_info[2], reconstruction_info[1], reconstruction_info[0], [model_name],  all_images_names, out_dir)
 

def inference_evaluation(
    test_dir, 
    model_paths, 
    model_names,
    violin_plot_filename,
    dataset_name,
    patch_shape,
    patch_step,
    batch_size
): 
    all_images_names, dataset, reconstruction_info = create_dataset_inference(
        test_dir,
        patch_shape=patch_shape, 
        patch_step=patch_step,
    )
    
    predictions = []
    
    for model_path in model_paths: 
        predictions.append(predict_model(dataset[0], model_path, batch_size))
    
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
    
    img_patches_matrix = create_matrix_images_as_rows_patches_as_cols(dataset[0], reconstruction_info[3])
    mask_patches_matrix = create_matrix_images_as_rows_patches_as_cols(dataset[1], reconstruction_info[3])
        
    image_slider = IntSlider(
        value=0,
        min=0,
        max= len(img_patches_matrix) - 1,
        step=1,
        description='Current Image: ',
        continuous_update=False
    )
    
    interact(visualize_reconstructed_images, images=fixed(img_patches_matrix), masks=fixed(mask_patches_matrix), predictions=fixed(predictions), model_names=fixed(model_names), nonreshaped_patches_arr_sizes=fixed(reconstruction_info[2]), reshaped_patches_arr_sizes=fixed(reconstruction_info[3]), nonpadded_image_sizes=fixed(reconstruction_info[0]), padded_image_sizes=fixed(reconstruction_info[1]), dataset_name=fixed(dataset_name), image_index=image_slider)
    
    #print('Saving inference masks...')
    
    #save_masks_inference(predictions, reconstruction_info[3], reconstruction_info[2], reconstruction_info[1], reconstruction_info[0], dataset_name, model_names, all_images_names)

    print('Getting stats...')
    plot_violin(predictions, dataset[1], model_names, violin_plot_filename)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    