import torch

def noop(**kwargs):
    pass

def run_ddpm(net, prompts, images=None, iterations=20, callback=noop, device="cpu"):
    if images == None:
        cur_images = torch.rand((1,3,640,480))
    else:
        cur_images = torch.rand_like(images)
    
    cur_images = cur_images.to(device)
    for i in range(iterations):
        cur_images = net(cur_images, prompts)
    return cur_images