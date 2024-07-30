import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = None
def log_data(data, i):

    
    for key in data.keys():
        writer.add_scalar(key, data[key], i)

def log_img(img, name):
    writer.add_image(name, img)



def init_logger(net, imgs=None, dir="runs"):
    net.eval()
    global writer
    if not writer:
        writer = SummaryWriter(dir)
    if imgs != None:
        writer.add_graph(net, imgs)
        writer.close()
    net.train()


def select_n_random(data, n=100):

    perm = torch.randperm(len(data))
    return data[perm][:n]

def add_projector(latents, images, global_step=0):
    features = latents

    writer.add_embedding(features,
    label_img=images,global_step=global_step)
    #writer.close()

def cache_break():
    writer.add_image('image', np.ones((3,3,3)), 0)


def log_embedding(dataset, model, n=100):
    with torch.no_grad():
        points = select_n_random(dataset, n)
        print(points.shape)

        latents = model.encode(points)

        add_projector(points, latents)