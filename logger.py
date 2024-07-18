import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

writer = None
def log_data(data, i):

    
    for key in data.keys():
        writer.add_scalar(key, data[key], i)

def log_img(img, name):
    writer.add_image(name, img)


def save_grid_with_label(img_grid, label, out_file):
    img_grid = img_grid.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_grid)
    ax.set_title(label, fontsize=20)
    ax.axis('off')


    plt.subplots_adjust(top=0.85)

    plt.savefig(out_file, bbox_inches='tight', pad_inches=0.1)


    plt.close(fig)
    plt.close("all")




def init_logger(dir="runs"):

    global writer
    if not writer:
        writer = SummaryWriter(dir)
    