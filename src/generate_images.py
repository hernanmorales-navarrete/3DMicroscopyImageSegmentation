# Import necessary libraries
import os  # For operating system dependent functionality (e.g., file paths)
import glob  # For file pattern matching (e.g., finding files with specific extensions)
import logging  # For logging messages (useful for tracking and debugging)
import numpy as np  # For numerical operations (e.g., arrays, random number generation)
import matplotlib.pyplot as plt  # For plotting images and visualizing data
import tifffile as tiff  # For reading and writing TIFF image files
from tqdm import tqdm  # For progress bar functionality (useful for tracking long loops)
from scipy.signal import fftconvolve  # For convolution using FFT (Fast Fourier Transform)
from scipy.ndimage import convolve  # For multi-dimensional convolution
from scipy.ndimage import zoom

# Set a random seed for reproducibility
np.random.seed(42)


# Define Functions


def load_image(image_path, plotImage=False):
    """
    Load a 3D TIFF (z-stack) image.
    """
    # Load the 3D TIFF (z-stack) image
    image = tiff.imread(image_path)
    # Check the shape of the image (should be something like (z, height, width))
    # print_image_features(image)
    if plotImage:
        # Visualize the middle slice of the z-stack
        visualize_middle_slice(image)
    return image


def print_image_features(image):
    """
    Print the shape and data type of the image.
    """
    print("Image shape:", image.shape)
    print("Image data type:", image.dtype)
    print("Image min value:", np.min(image))
    print("Image max value:", np.max(image))


def visualize_middle_slice(zstack_image):
    """
    Visualize the middle slice of the z-stack.
    """
    middle_slice = zstack_image[zstack_image.shape[0] // 2]
    plt.imshow(middle_slice, cmap="gray")
    plt.axis("off")
    plt.show()


def normalize_psf(psf):
    """
    Check if the PSF is normalized (i.e., the sum of all elements equals 1).
    If not, normalize the PSF by dividing each element by the sum of all elements.
    """
    psf_sum = np.sum(psf)
    if not np.isclose(psf_sum, 1.0):  # Check if the sum is close to 1
        print(f"PSF is not normalized. Normalizing now. Sum of PSF = {psf_sum}")
        psf /= psf_sum  # Normalize the PSF
    else:
        print(f"PSF is already normalized. Sum of PSF = {psf_sum}")

    return psf


# Uneven-staining
def get_smooth_variations(shape, binning, scale=5):
    # get dimensions
    small_shape = tuple(dim // binning for dim in shape)
    zoom_factors = [orig_dim / small_dim for orig_dim, small_dim in zip(shape, small_shape)]
    # Generate local_variation with half the shape
    local_variation = np.random.normal(size=small_shape) + 1.0
    # Apply Gaussian smoothing for downscale frequency to simulate smooth variations
    smoothed_variation_temp = convolve(local_variation, np.ones((scale, scale, scale)) / scale**3)
    # Extrapolate somoothed_variation to the original shape
    smoothed_variation = zoom(
        smoothed_variation_temp, zoom_factors, order=1
    )  # order=1 for bilinear interpolation

    return smoothed_variation


# Function to simulate local variations in staining (Gaussian random field)
def simulate_local_variations(shape, binnings=[1, 2, 4, 8], scale=5):
    # Initialize the result array with ones in the desired shape
    result = np.ones(shape)
    # Loop over each scale factor in binning scales to generate local variations
    for s in binnings:
        smoothed_variation = get_smooth_variations(shape, s, scale)
        result *= smoothed_variation  # Element-wise multiplication
    return result


# Function to simulate reduction in staining penetration along the z-axis
def simulate_staining_penetration(binary_mask, z_decay_rate=0.999):
    z_stack_depth = binary_mask.shape[0]
    penetration_profile = np.exp(-np.arange(z_stack_depth) * (1 - z_decay_rate))[:, None, None]
    return binary_mask * penetration_profile


# Function to combine both local variations and depth-dependent staining penetration
def simulate_uneven_staining(binary_mask, local_variation_scale=5, z_decay_rate=0.999):
    # Simulate local variations
    local_variation = simulate_local_variations(
        binary_mask.shape, binnings=[1, 2, 4, 8], scale=local_variation_scale
    )
    # save_image(local_variation, "local_variation.tif", out_dir_images, 32)

    # Simulate staining penetration reduction
    if z_decay_rate < 1 and z_decay_rate > 0:
        stained_image = simulate_staining_penetration(binary_mask, z_decay_rate=z_decay_rate)
    else:
        stained_image = binary_mask

    # save_image(stained_image, "degradation.tif", out_dir_images)
    # Combine both effects: local variations + depth penetration
    res = stained_image * local_variation  # Adding local variations to the staining profile
    res[res < 0] = 0

    return res


# Add BG
def add_BG(stained_image, background_level, image_intensity_scale, local_variation_scale=5):
    BG = (
        np.ones(stained_image.shape) * background_level * image_intensity_scale
    )  # Contsnat background
    BG_noisy = BG * simulate_local_variations(
        stained_image.shape, binnings=[1, 2], scale=local_variation_scale
    )  # add local noise
    BG_noisy[stained_image != 0] = 0  # set 0 to places with FG
    BG_noisy[BG_noisy < 0] = 0
    # save_image(BG_noisy, "BG_noisy.tif", out_dir_images)

    return stained_image + BG_noisy


# Convolution
def convolve_with_psf(image_3D, psf):
    """
    Convolve the 3D image stack with the PSF.
    """
    # Nomrmalize psf
    psf = normalize_psf(psf)

    # Calculate the required padding (half the size of the PSF in each dimension)
    pad_size = psf.shape  # [size // 2 for size in psf.shape] + 1

    # Pad the image with 'reflect' or other padding mode to reduce artifacts
    padded_image = np.pad(image_3D, [(p, p) for p in pad_size], mode="reflect")

    # Convolve the 3D image stack with the PSF
    convolved_padded = fftconvolve(padded_image, psf, mode="same")

    # Remove the padding to restore the original image size
    slices = tuple(slice(p, -p) for p in pad_size)
    convolved_image = convolved_padded[slices]

    # Set negative values to 0
    convolved_image[convolved_image < 0] = 0
    return convolved_image


# Function to add Poisson noise
def add_poisson_noise(image, scaling_factor=1.0):
    scaled_image = image * scaling_factor
    noisy_image = np.random.poisson(scaled_image)
    noisy_poisson_image = noisy_image / scaling_factor
    noisy_poisson_image[noisy_poisson_image < 0] = 0
    return noisy_poisson_image


# Function to add Gaussian noise
def add_gaussian_noise(image, std_dev=1.0):
    noise = np.random.normal(0, std_dev, image.shape)
    noisy_image = image + noise
    noisy_image[noisy_image < 0] = 0
    return noisy_image


def calculate_snr(signal, noise, mask=None):
    if mask is not None:
        signal = signal * mask
    # Consider only FG values for mean
    non_zero_values = signal[signal != 0]
    signal_mean = np.mean(non_zero_values)
    # Calculate noise std
    noise_std = np.std(noise)
    # print(f"Signal mean: {signal_mean}, Noise std: {noise_std}")
    return signal_mean / noise_std


# Function to save the image
def save_image(image, filename, out_dir, bit_depth=16):
    filepath = os.path.join(out_dir, filename)
    # Ensure the image data is cast to the desired bit depth
    if bit_depth == 16:
        image = image.astype(np.uint16)  # Convert to 16-bit unsigned integer
    elif bit_depth == 8:
        image = image.astype(np.uint8)  # Convert to 8-bit unsigned integer
    elif bit_depth == 32:
        image = image.astype(np.float32)  # Convert to 32-bit floating point
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    # Save the 3D image as a TIFF file
    tiff.imwrite(filepath, image)


def generate_SNR_images(
    noisy_poisson_image,
    convolved_image,
    binary_mask,
    snr_targets,
    snr_tolerance,
    std_dev,
    max_iterations,
    basename,
    out_dir_masks,
    out_dir_images,
):
    # Get Images for all SNR
    for snr_target in snr_targets:
        iteration = 0
        snr = 0
        with tqdm(total=max_iterations, desc="Iterations", unit="iteration") as pbar:
            while iteration < max_iterations:
                # Add Gaussian noise with the current std_dev
                final_image = add_gaussian_noise(noisy_poisson_image, std_dev=std_dev)

                # Calculate noise (difference from convolved image) and SNR
                noise = final_image - convolved_image
                snr = calculate_snr(convolved_image, noise, binary_mask)

                # Check if we are close to the target SNR
                if np.abs(snr - snr_target) < snr_tolerance or std_dev < 1e-6:
                    filename = f"{basename}_{snr_target}_SNR.tif"
                    save_image(final_image, filename, out_dir_images)
                    save_image(binary_mask, filename, out_dir_masks)
                    break

                # Adjust std_dev to bring SNR closer to the target
                if snr < snr_target:
                    std_dev *= 0.9  # Reduce noise to increase SNR
                else:
                    std_dev *= 1.1  # Increase noise to decrease SNR

                iteration += 1

                # Update progress bar and print progress
                pbar.update(1)
                pbar.set_postfix({"SNR": f"{snr:.2f}", "std_dev": f"{std_dev:.4f}"})

        print(
            f" Final SNR: {snr:.2f} (Target SNR: {snr_target}), Gaussian std_dev: {std_dev:.4f}, Iterations: {iteration}"
        )  # Define Functions


def load_image(image_path, plotImage=False):
    """
    Load a 3D TIFF (z-stack) image.
    """
    # Load the 3D TIFF (z-stack) image
    image = tiff.imread(image_path)
    # Check the shape of the image (should be something like (z, height, width))
    # print_image_features(image)
    if plotImage:
        # Visualize the middle slice of the z-stack
        visualize_middle_slice(image)
    return image


def print_image_features(image):
    """
    Print the shape and data type of the image.
    """
    print("Image shape:", image.shape)
    print("Image data type:", image.dtype)
    print("Image min value:", np.min(image))
    print("Image max value:", np.max(image))


def visualize_middle_slice(zstack_image):
    """
    Visualize the middle slice of the z-stack.
    """
    middle_slice = zstack_image[zstack_image.shape[0] // 2]
    plt.imshow(middle_slice, cmap="gray")
    plt.axis("off")
    plt.show()


def normalize_psf(psf):
    """
    Check if the PSF is normalized (i.e., the sum of all elements equals 1).
    If not, normalize the PSF by dividing each element by the sum of all elements.
    """
    psf_sum = np.sum(psf)
    if not np.isclose(psf_sum, 1.0):  # Check if the sum is close to 1
        print(f"PSF is not normalized. Normalizing now. Sum of PSF = {psf_sum}")
        psf /= psf_sum  # Normalize the PSF
    else:
        print(f"PSF is already normalized. Sum of PSF = {psf_sum}")

    return psf


# Uneven-staining
def get_smooth_variations(shape, binning, scale=5):
    # get dimensions
    small_shape = tuple(dim // binning for dim in shape)
    zoom_factors = [orig_dim / small_dim for orig_dim, small_dim in zip(shape, small_shape)]
    # Generate local_variation with half the shape
    local_variation = np.random.normal(size=small_shape) + 1.0
    # Apply Gaussian smoothing for downscale frequency to simulate smooth variations
    smoothed_variation_temp = convolve(local_variation, np.ones((scale, scale, scale)) / scale**3)
    # Extrapolate somoothed_variation to the original shape
    smoothed_variation = zoom(
        smoothed_variation_temp, zoom_factors, order=1
    )  # order=1 for bilinear interpolation

    return smoothed_variation


# Function to simulate local variations in staining (Gaussian random field)
def simulate_local_variations(shape, binnings=[1, 2, 4, 8], scale=5):
    # Initialize the result array with ones in the desired shape
    result = np.ones(shape)
    # Loop over each scale factor in binning scales to generate local variations
    for s in binnings:
        smoothed_variation = get_smooth_variations(shape, s, scale)
        result *= smoothed_variation  # Element-wise multiplication
    return result


# Function to simulate reduction in staining penetration along the z-axis
def simulate_staining_penetration(binary_mask, z_decay_rate=0.999):
    z_stack_depth = binary_mask.shape[0]
    penetration_profile = np.exp(-np.arange(z_stack_depth) * (1 - z_decay_rate))[:, None, None]
    return binary_mask * penetration_profile


# Function to combine both local variations and depth-dependent staining penetration
def simulate_uneven_staining(binary_mask, local_variation_scale=5, z_decay_rate=0.999):
    # Simulate local variations
    local_variation = simulate_local_variations(
        binary_mask.shape, binnings=[1, 2, 4, 8], scale=local_variation_scale
    )
    # save_image(local_variation, "local_variation.tif", out_dir_images, 32)

    # Simulate staining penetration reduction
    if z_decay_rate < 1 and z_decay_rate > 0:
        stained_image = simulate_staining_penetration(binary_mask, z_decay_rate=z_decay_rate)
    else:
        stained_image = binary_mask

    # save_image(stained_image, "degradation.tif", out_dir_images)
    # Combine both effects: local variations + depth penetration
    res = stained_image * local_variation  # Adding local variations to the staining profile
    res[res < 0] = 0

    return res


# Add BG
def add_BG(stained_image, background_level, image_intensity_scale, local_variation_scale=5):
    BG = (
        np.ones(stained_image.shape) * background_level * image_intensity_scale
    )  # Contsnat background
    BG_noisy = BG * simulate_local_variations(
        stained_image.shape, binnings=[1, 2], scale=local_variation_scale
    )  # add local noise
    BG_noisy[stained_image != 0] = 0  # set 0 to places with FG
    BG_noisy[BG_noisy < 0] = 0
    # save_image(BG_noisy, "BG_noisy.tif", out_dir_images)

    return stained_image + BG_noisy


# Convolution
def convolve_with_psf(image_3D, psf):
    """
    Convolve the 3D image stack with the PSF.
    """
    # Nomrmalize psf
    psf = normalize_psf(psf)

    # Calculate the required padding (half the size of the PSF in each dimension)
    pad_size = psf.shape  # [size // 2 for size in psf.shape] + 1

    # Pad the image with 'reflect' or other padding mode to reduce artifacts
    padded_image = np.pad(image_3D, [(p, p) for p in pad_size], mode="reflect")

    # Convolve the 3D image stack with the PSF
    convolved_padded = fftconvolve(padded_image, psf, mode="same")

    # Remove the padding to restore the original image size
    slices = tuple(slice(p, -p) for p in pad_size)
    convolved_image = convolved_padded[slices]

    # Set negative values to 0
    convolved_image[convolved_image < 0] = 0
    return convolved_image


# Function to add Poisson noise
def add_poisson_noise(image, scaling_factor=1.0):
    scaled_image = image * scaling_factor
    noisy_image = np.random.poisson(scaled_image)
    noisy_poisson_image = noisy_image / scaling_factor
    noisy_poisson_image[noisy_poisson_image < 0] = 0
    return noisy_poisson_image


# Function to add Gaussian noise
def add_gaussian_noise(image, std_dev=1.0):
    noise = np.random.normal(0, std_dev, image.shape)
    noisy_image = image + noise
    noisy_image[noisy_image < 0] = 0
    return noisy_image


def calculate_snr(signal, noise, mask=None):
    if mask is not None:
        signal = signal * mask
    # Consider only FG values for mean
    non_zero_values = signal[signal != 0]
    signal_mean = np.mean(non_zero_values)
    # Calculate noise std
    noise_std = np.std(noise)
    # print(f"Signal mean: {signal_mean}, Noise std: {noise_std}")
    return signal_mean / noise_std


# Function to save the image
def save_image(image, filename, out_dir, bit_depth=16):
    filepath = os.path.join(out_dir, filename)
    # Ensure the image data is cast to the desired bit depth
    if bit_depth == 16:
        image = image.astype(np.uint16)  # Convert to 16-bit unsigned integer
    elif bit_depth == 8:
        image = image.astype(np.uint8)  # Convert to 8-bit unsigned integer
    elif bit_depth == 32:
        image = image.astype(np.float32)  # Convert to 32-bit floating point
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    # Save the 3D image as a TIFF file
    tiff.imwrite(filepath, image)


def generate_SNR_images(
    noisy_poisson_image,
    convolved_image,
    binary_mask,
    snr_targets,
    snr_tolerance,
    std_dev,
    max_iterations,
    basename,
    out_dir_masks,
    out_dir_images,
):
    # Get Images for all SNR
    for snr_target in snr_targets:
        iteration = 0
        snr = 0
        with tqdm(total=max_iterations, desc="Iterations", unit="iteration") as pbar:
            while iteration < max_iterations:
                # Add Gaussian noise with the current std_dev
                final_image = add_gaussian_noise(noisy_poisson_image, std_dev=std_dev)

                # Calculate noise (difference from convolved image) and SNR
                noise = final_image - convolved_image
                snr = calculate_snr(convolved_image, noise, binary_mask)

                # Check if we are close to the target SNR
                if np.abs(snr - snr_target) < snr_tolerance or std_dev < 1e-6:
                    filename = f"{basename}_{snr_target}_SNR.tif"
                    save_image(final_image, filename, out_dir_images)
                    save_image(binary_mask, filename, out_dir_masks)
                    break

                # Adjust std_dev to bring SNR closer to the target
                if snr < snr_target:
                    std_dev *= 0.9  # Reduce noise to increase SNR
                else:
                    std_dev *= 1.1  # Increase noise to decrease SNR

                iteration += 1

                # Update progress bar and print progress
                pbar.update(1)
                pbar.set_postfix({"SNR": f"{snr:.2f}", "std_dev": f"{std_dev:.4f}"})

        print(
            f" Final SNR: {snr:.2f} (Target SNR: {snr_target}), Gaussian std_dev: {std_dev:.4f}, Iterations: {iteration}"
        )


def simulate_images(
    scr_dir,
    psf_path,
    out_dir,
    snr_targets,
    background_level,
    local_variation_scale,
    z_decay_rate,
    std_dev,
    max_iterations=200,  # Maximum number of iterations for optimization
    snr_tolerance=0.1,  # Maximum tolerance for teh calculation of the SNR
    image_intensity_scale=1000,
    poisson_scaling=1.0,
    save_input=True,
):
    # Create output folders
    scr_dir_masks = scr_dir + "masks/"
    scr_dir_images = scr_dir + "images/"
    out_dir_masks = out_dir + "masks/"
    out_dir_images = out_dir + "images/"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_dir_masks):
        os.makedirs(out_dir_masks)
    if not os.path.exists(out_dir_images):
        os.makedirs(out_dir_images)

    # Load the PSF image
    psf = load_image(psf_path)
    print(f"PSF Loaded: {psf_path}")

    # Use glob to get all .tif image files in the directory
    image_files = glob.glob(os.path.join(scr_dir_images, "*.tif"))
    masks_files = glob.glob(os.path.join(scr_dir_masks, "*.tif"))

    # Sort the list from largest to smallest
    snr_targets = sorted(snr_targets, reverse=True)

    # Loop through each image file
    for img_path, mask_path in zip(image_files, masks_files):
        print(f"Processing file: {mask_path}")

        # Get image name
        file_name = os.path.splitext(os.path.basename(mask_path))[0]

        # Load the 3D TIFF (z-stack) image
        binary_mask = load_image(mask_path)
        binary_mask = binary_mask.astype(np.uint16)
        binary_mask[binary_mask > 0] = 1  # make it binary
        image = binary_mask * image_intensity_scale  # Sacle intensity

        if save_input:
            img = load_image(img_path)
            save_image(binary_mask, file_name + ".tif", out_dir_masks)
            save_image(img, file_name + ".tif", out_dir_images)

        # Step 1: Generate uneven staining pattern (e.g., simple gradient)
        stained_image = simulate_uneven_staining(
            image, local_variation_scale=local_variation_scale, z_decay_rate=z_decay_rate
        )
        # save_image(stained_image, "staining.tif", out_dir_images)

        # Step 2 : Add BG
        stained_image = add_BG(
            stained_image, background_level, image_intensity_scale, local_variation_scale=5
        )
        # stained_image[stained_image == 0] = background_level  * image_intensity_scale  # Add background
        # save_image(stained_image, "image_BG.tif", out_dir_images)

        # Step 3: Convolve with provided 3D PSF
        convolved_image = convolve_with_psf(stained_image, psf)
        # save_image(convolved_image, "convolved_image.tif", out_dir_images)

        # Step 4: Add Poisson noise
        noisy_poisson_image = add_poisson_noise(convolved_image, poisson_scaling)
        # save_image(noisy_poisson_image, "noisy_poisson_image.tif", out_dir_images)

        # Step 5: Genrate images with diffrentes SNR
        generate_SNR_images(
            noisy_poisson_image,
            convolved_image,
            binary_mask,
            snr_targets,
            snr_tolerance,
            std_dev,
            max_iterations,
            file_name,
            out_dir_masks,
            out_dir_images,
        )
