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
    pad_height = (
        math.ceil((image.shape[0] - patch_shape[0]) / step) * step + patch_shape[0]
    ) - image.shape[0]
    pad_width = (
        math.ceil((image.shape[1] - patch_shape[1]) / step) * step + patch_shape[1]
    ) - image.shape[1]
    pad_depth = (
        math.ceil((image.shape[2] - patch_shape[2]) / step) * step + patch_shape[2]
    ) - image.shape[2]

    padded_image = np.pad(
        image, ((0, pad_height), (0, pad_width), (0, pad_depth)), mode="constant"
    )
    return padded_image


def patchify3DImage(source_img, patch_shape, patch_step):
    img_patches = []

    img = tiff.imread(source_img)
    nonpadded_img_size = img.shape

    img = pad_image(img, patch_shape, patch_step)
    padded_img_size = img.shape

    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    patches = patchify(img, patch_size=patch_shape, step=patch_step)
    nonreshaped_patches_arr_size = patches.shape

    patches = np.reshape(patches, (-1, patch_shape[0], patch_shape[1], patch_shape[2]))
    reshaped_patches_arr_size = patches.shape

    for patch in patches:
        img_patches.append(np.expand_dims(patch, -1))

    return (
        img_patches,
        nonpadded_img_size,
        padded_img_size,
        nonreshaped_patches_arr_size,
        reshaped_patches_arr_size,
    )


def create_patches_from_images_in_dir(data_dir, patch_shape, patch_step):
    images_dir = Path(join(data_dir, "images"))
    masks_dir = Path(join(data_dir, "masks"))

    images_files = sorted(images_dir.glob("*.tif"))
    masks_files = sorted(masks_dir.glob("*.tif"))

    mask_dict = {mask.stem: mask for mask in masks_files}

    all_images_patches = []
    all_masks_patches = []

    nonpadded_image_sizes = []
    padded_image_sizes = []
    nonreshaped_patches_arr_sizes = []
    reshaped_patches_arr_sizes = []

    all_images_names = []

    for image in images_files:
        key = image.stem
        mask = mask_dict.get(key)

        print(f"Processing {image.stem} image and mask")

        (
            img_patches,
            nonpadded_img_size,
            padded_image_size,
            nonreshaped_patches_arr_size,
            reshaped_patches_arr_size,
        ) = patchify3DImage(image, patch_shape, patch_step)
        mask_patches, *ignored = patchify3DImage(mask, patch_shape, patch_step)

        all_images_patches.extend(img_patches)
        all_masks_patches.extend(mask_patches)

        nonpadded_image_sizes.append(nonpadded_img_size)
        padded_image_sizes.append(padded_image_size)
        nonreshaped_patches_arr_sizes.append(nonreshaped_patches_arr_size)
        reshaped_patches_arr_sizes.append(reshaped_patches_arr_size)

        all_images_names.append(image.stem)

    return (
        all_images_names,
        np.array(all_images_patches),
        np.array(all_masks_patches),
        nonpadded_image_sizes,
        padded_image_sizes,
        nonreshaped_patches_arr_sizes,
        reshaped_patches_arr_sizes,
    )


def create_patches_from_images_in_dir_only_images(input_dir, patch_shape, patch_step):
    images_dir = Path(input_dir)
    images_files = sorted(images_dir.glob("*.tif"))
    all_images_patches = []

    nonpadded_image_sizes = []
    padded_image_sizes = []
    nonreshaped_patches_arr_sizes = []
    reshaped_patches_arr_sizes = []

    all_images_names = []

    for image in images_files:
        key = image.stem
        print(f"Processing {image.stem} image")

        (
            img_patches,
            nonpadded_img_size,
            padded_image_size,
            nonreshaped_patches_arr_size,
            reshaped_patches_arr_size,
        ) = patchify3DImage(image, patch_shape, patch_step)

        all_images_patches.extend(img_patches)

        nonpadded_image_sizes.append(nonpadded_img_size)
        padded_image_sizes.append(padded_image_size)
        nonreshaped_patches_arr_sizes.append(nonreshaped_patches_arr_size)
        reshaped_patches_arr_sizes.append(reshaped_patches_arr_size)

        all_images_names.append(image.stem)

    return (
        all_images_names,
        np.array(all_images_patches),
        nonpadded_image_sizes,
        padded_image_sizes,
        nonreshaped_patches_arr_sizes,
        reshaped_patches_arr_sizes,
    )


def create_matrix_images_as_rows_patches_as_cols(patches, patches_per_images):
    patches_per_image = []
    start = 0
    for num_patches in patches_per_images:
        end = start + num_patches[0]
        patches_per_image.append(patches[start:end])
        start = end

    return patches_per_image


def create_dataset_inference(dir, patch_shape, patch_step):
    (
        all_images_names,
        all_images_patches,
        all_masks_patches,
        nonpadded_image_sizes,
        padded_image_sizes,
        nonreshaped_patches_arr_sizes,
        reshaped_patches_arr_sizes,
    ) = create_patches_from_images_in_dir(dir, patch_shape, patch_step)

    dataset = (all_images_patches, all_masks_patches)
    reconstruction_info = (
        nonpadded_image_sizes,
        padded_image_sizes,
        nonreshaped_patches_arr_sizes,
        reshaped_patches_arr_sizes,
    )

    return all_images_names, dataset, reconstruction_info


def create_dataset_prediction(dir, patch_shape, patch_step):
    (
        all_images_names,
        all_images_patches,
        nonpadded_image_sizes,
        padded_image_sizes,
        nonreshaped_patches_arr_sizes,
        reshaped_patches_arr_sizes,
    ) = create_patches_from_images_in_dir_only_images(dir, patch_shape, patch_step)

    dataset = all_images_patches
    reconstruction_info = (
        nonpadded_image_sizes,
        padded_image_sizes,
        nonreshaped_patches_arr_sizes,
        reshaped_patches_arr_sizes,
    )

    return all_images_names, dataset, reconstruction_info
