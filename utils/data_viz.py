import matplotlib.pyplot as plt

def visualize_patch_3D_in_2D(image, mask, z):
    img_2D = image[:, :, z]
    mask_2D = mask[:, :, z]

    fig, axes = plt.subplots(1, 2, figsize=(6, 6))

    axes[0].imshow(img_2D, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Image')

    axes[1].imshow(mask_2D, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Mask')

    plt.tight_layout()
    plt.show()
