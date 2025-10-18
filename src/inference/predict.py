from pathlib import Path

from numpy.typing import NDArray
import tifffile

from src.config import ALLOWED_EXTENSIONS, CLASSICAL_METHODS, MAX_WORKERS
from src.inference.utils import apply_classical_thresholding_and_save_masks_for_array_of_filenames, extract_patch_info
from src.utils import create_directory


class Prediction:
    """
    This class performs predictions at a patch-level and image level. 
    patch-level: each patch's prediction is processed
    image-level: each patch's prediction is reconstructed and saved 

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
    def __init__(self, patch_dir: Path, output_dir: Path, dataset_name: Path):
        #Get subdirectories correspoding to image names, each subdir contains patches of a single image
        self.subdirs_corresponding_to_volume_names = [d for d in patch_dir.glob("*/")]

        #Define where to store image-level predictions and patch-level predictions
        self.directory_for_predictions = output_dir / dataset_name
        self.directory_for_image_level_predictions = self.directory_for_predictions / "image_level"
        self.directory_for_patch_level_predictions = self.directory_for_predictions / "patch_level"
        
        create_directory(self.directory_for_image_level_predictions)
        create_directory(self.directory_for_patch_level_predictions)

        if not self.subdirs_corresponding_to_volume_names: 
            raise Exception("No subdirectories in patch_dir")
        
        #Check that ALL subdirectories' files have valid file extesions
        for subdir in self.subdirs_corresponding_to_volume_names:
            if any(f.suffix.lower() not in ALLOWED_EXTENSIONS for f in subdir.glob("*")):
                raise Exception(f"Invalid extensions in {subdir.name}")
    
    def predict_and_save_at_patch_level(self):
        for subdir in self.subdirs_corresponding_to_image_names:
            # Obtain image's patches filenames and sort them by ID
            # IMPORTANT: THE IMAGES SHOULD BE SORTED! 
            image_patches_filenames = sorted(subdir.glob("*"), key=lambda filename: extract_patch_info(filename)[4])

            #Predict patches with classical thresholding methods
            for method in CLASSICAL_METHODS: 
                apply_classical_thresholding_and_save_masks_for_array_of_filenames(image_patches_filenames, self.directory_for_patch_level_predictions, method, MAX_WORKERS)
            
            #Predict complete images (image-level) with classical thresholding methods
        
            #Predict patches with deep learning methods. This includes image-level and patch-level predictions; remember, we reconstruct predictions for patches in order to build the whole image
    
    






