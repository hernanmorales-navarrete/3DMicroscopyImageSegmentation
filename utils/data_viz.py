import matplotlib.pyplot as plt

def visualize_patch_3D_in_2D(image, mask, prediction, title, z):
    img_2D = image[:, :, z]
    mask_2D = mask[:, :, z]
    pred_2D = prediction[:, :, z]

    fig, axes = plt.subplots(1, 3, figsize=(6, 6))

    axes[0].imshow(img_2D, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Image')

    axes[1].imshow(mask_2D, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Mask')
    
    axes[2].imshow(pred_2D, cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Prediction')
    
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
