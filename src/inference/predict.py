from pathlib import Path

from loguru import logger

from src.config import ALLOWED_EXTENSIONS, BATCH_SIZE, CLASSICAL_METHODS, MAX_WORKERS, PATCH_SIZE
from src.inference.utils import (
    apply_classical_thresholding_and_save_masks_for_array_of_filenames,
    apply_deep_learning_method_to_array_of_filenames,
    extract_patch_info,
    get_deep_learning_models_from_dir,
)
from src.utils import create_directory


class Prediction:
    """
    This class performs predictions for classical models and deep learning models: 
    1. For classical models, we take an array of patches

    The task's result is a mask. 

    In both cases, the results are stored in different folders: patch-level and image-level. Each directory inside patch-level and image-level will have folders with the names of prediction methods and within them the actual predictions. 

    MY_DATASET
    |   patch_level
    |   |   image_name
    |      |   otsu
    |      |   frangi
    |      |

    The user can define where it stores the predictions and how the directory containing all the predictions is called
    """
    def __init__(self, patch_dir: Path, images_dir: Path, output_dir: Path, dataset_name: Path, models_dir: Path):
        #Get subdirectories correspoding to image names, each subdir contains patches of a single image
        self.subdirs_corresponding_to_image_names = [d for d in patch_dir.glob("*/")]
        self.images_dir = images_dir

        #Get deep learning models from dir
        self.models_list = get_deep_learning_models_from_dir(models_dir)

        #Define where to store image-level predictions and patch-level predictions
        self.directory_for_predictions = output_dir / dataset_name
        self.directory_for_image_level_predictions = self.directory_for_predictions / "image_level"
        self.directory_for_patch_level_predictions = self.directory_for_predictions / "patch_level"

        if not self.subdirs_corresponding_to_image_names: 
            raise Exception("No subdirectories in patch_dir")
        
        if any(f.suffix.lower() not in ALLOWED_EXTENSIONS for f in self.images_dir.glob("*")): 
            raise Exception(f"There are invalid extensions in {images_dir}")
        
        #Check that ALL subdirectories' files have valid file extesions
        for subdir in self.subdirs_corresponding_to_image_names:
            if any(f.suffix.lower() not in ALLOWED_EXTENSIONS for f in subdir.glob("*")):
                raise Exception(f"Invalid extensions in {subdir.name}")

        create_directory(self.directory_for_image_level_predictions)
        create_directory(self.directory_for_patch_level_predictions)
    
    def predict(self):
        for subdir in self.subdirs_corresponding_to_image_names:
            # Obtain image's patches filenames and sort them by ID
            # IMPORTANT: THE IMAGES SHOULD BE SORTED! 
            image_patches_absolute_paths = sorted(subdir.glob("*"), key=lambda filename: extract_patch_info(filename)[4])

            #Get the corresponding image from images folder
            complete_image_filename = self.images_dir / subdir.name

            #Predict patches with classical thresholding methods
            logger.info("Applying classical thresholding for patches and complete images and saving masks")
            for method in CLASSICAL_METHODS: 
                apply_classical_thresholding_and_save_masks_for_array_of_filenames(image_patches_absolute_paths, self.directory_for_patch_level_predictions, method, MAX_WORKERS)
                apply_classical_thresholding_and_save_masks_for_array_of_filenames([complete_image_filename], self.directory_for_image_level_predictions, method, MAX_WORKERS)
        
            #Predict patches with deep learning methods. This includes image-level and patch-level predictions; remember, we reconstruct predictions for patches in order to build the whole image
            logger.info("Applying deep learning methods for patches, saving them, reconstructing the patches and saving the entire image")
            for model_name, augmentation, best_model_path in self.models_list: 
                apply_deep_learning_method_to_array_of_filenames(image_patches_absolute_paths, self.directory_for_patch_level_predictions, self.directory_for_image_level_predictions, model_name, augmentation, best_model_path, BATCH_SIZE, PATCH_SIZE, MAX_WORKERS)