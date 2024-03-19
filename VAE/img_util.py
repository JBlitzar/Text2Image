import matplotlib.pyplot as plt
import numpy as np

def save_side_by_side_image(image1, image2, save_path):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image1)
    axes[0].axis('off')
    axes[1].imshow(image2)
    axes[1].axis('off')
    plt.savefig(save_path)
    plt.close()
