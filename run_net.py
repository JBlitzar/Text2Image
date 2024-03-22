import torch
import torchvision.transforms.v2 as v2
def noop(**kwargs):
    pass

def run_ddpm(net, prompts, images=None, iterations=10, callback=noop, device="cpu"):
    if images == None:
        cur_images = torch.rand((1,3,640,480))
    else:
        cur_images = torch.rand_like(images)
    
    cur_images = cur_images.to(device)
    for i in range(iterations):
        cur_images = net(cur_images, prompts)
    return cur_images, images

def run_naive(net, prompts, images=None, device="cpu", callback=noop, *argv):
    results = net(prompts)
    images = v2.functional.crop(images, 0,80,480,480)
    return results, images