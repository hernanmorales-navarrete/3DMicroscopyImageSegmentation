from patchify import patchify
import cv2
import tifffile as tiff
import numpy as np
from glob import glob
from os.path import join
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
import tensorflow as tf



def patchify3DImage(source_img, patch_shape=(64, 64, 64), patch_step=64):
    
    img_patches = []
    img = tiff.imread(source_img)
    img = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    patches = patchify(img, patch_size = patch_shape, step = patch_step)
    patches = np.reshape(patches, (-1, patch_shape[0], patch_shape[1], patch_shape[2]))

    for patch in patches:
        img_patches.append(np.expand_dims(patch, -1))
        
    return img_patches

def create_patches_from_images_in_dir(data_dir, patch_shape=(64, 64, 64), patch_step=64):
    images_dir = Path(join(data_dir, 'images'))
    masks_dir = Path(join(data_dir, 'masks'))

    images_files = sorted(images_dir.glob('*.tif'))
    masks_files = sorted(masks_dir.glob('*.tif'))

    mask_dict = {mask.stem: mask for mask in masks_files}

    images = []
    masks = []

    for image in images_files: 
        key = image.stem
        mask = mask_dict.get(key)

        print(f'Processing {image.stem} image and mask')

        img_patches = patchify3DImage(image, patch_shape, patch_step)
        mask_patches = patchify3DImage(mask, patch_shape, patch_step)
        
        images.extend(img_patches)
        masks.extend(mask_patches)
    return np.array(images), np.array(masks)


def create_tf_datasets(dir, percent_val=None, patch_shape, patch_step, random_state=42): 
    patches, masks = create_patches_from_images_in_dir(dir, patch_shape, patch_step)
    
    if percent_val is not None: 
        train_patches, val_patches, train_masks, val_masks = train_test_split(patches, masks, test_size=percent_val, random_state=random_state)
        train_dataset = (train_patches, train_masks)
        val_dataset = (val_patches, val_masks)
        
        return train_dataset, val_dataset
    else:
        throw ValueError('Please provide a percentage for the validation dataset')
    
