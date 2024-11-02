from patchify import patchify
import cv2
import tifffile as tiff
import numpy as np
from glob import glob
from os.path import join
import os
from pathlib import Path
import math

from sklearn.model_selection import train_test_split
import tensorflow as tf

def pad_image(image, patch_shape, step):
    pad_height = (math.ceil((image.shape[0] - patch_shape[0]) / step) * step + patch_shape[0]) - image.shape[0]
    pad_width = (math.ceil((image.shape[1] - patch_shape[1]) / step) * step + patch_shape[1]) - image.shape[1]
    pad_depth = (math.ceil((image.shape[2] - patch_shape[2]) / step) * step + patch_shape[2]) - image.shape[2]
    
    padded_image = np.pad(
        image,
        (
            (0, pad_height),
            (0, pad_width),
            (0, pad_depth)
        ),
        mode='constant'
    )
    return padded_image


def patchify3DImage(source_img, patch_shape, patch_step):
    patches_from_image = []
    img = tiff.imread(source_img)
    img = pad_image(img, patch_shape, patch_step)
    img = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    patches = patchify(img, patch_size = patch_shape, step = patch_step)
    original_patches_shape = patches.shape
    patches = np.reshape(patches, (-1, patch_shape[0], patch_shape[1], patch_shape[2]))

    for patch in patches:
        patches_from_image.append(np.expand_dims(patch, -1))
    
    return patches_from_image, img.shape, original_patches_shape

def create_patches_from_images_in_dir(data_dir, patch_shape, patch_step):
    images_dir = Path(join(data_dir, 'images'))
    masks_dir = Path(join(data_dir, 'masks'))

    images_files = sorted(images_dir.glob('*.tif'))
    masks_files = sorted(masks_dir.glob('*.tif'))

    mask_dict = {mask.stem: mask for mask in masks_files}

    patches = []
    masks = []
    
    images_size = []
    patches_per_image = []
    patches_size = []

    for image in images_files: 
        key = image.stem
        mask = mask_dict.get(key)

        print(f'Processing {image.stem} image and mask')

        img_patches, img_shape, patches_shape = patchify3DImage(image, patch_shape, patch_step)
        mask_patches, mask_shape, mask_patches_shape = patchify3DImage(mask, patch_shape, patch_step)
        
        patches.extend(img_patches)
        masks.extend(mask_patches)
        
        images_size.append(img_shape)
        patches_per_image.append(len(img_patches))
        patches_size.append(patches_shape)
        
    return np.array(patches), np.array(masks), images_size, patches_size, patches_per_image

def create_array_patches_per_image(patches, patches_per_images):
    patches_per_image = []
    start = 0
    for num_patches in patches_per_images:
        end = start + num_patches
        patches_per_image.append(patches[start:end])
        start = end
        
    return patches_per_image


def create_dataset_inference(dir, patch_shape, patch_step): 
    patches, masks, images_size, patches_size ,patches_per_image = create_patches_from_images_in_dir(dir, patch_shape, patch_step)
    dataset = (patches, masks)
    reconstruction_info = (images_size, patches_size, patches_per_image)
    
    return dataset, reconstruction_info