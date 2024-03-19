from vae_architecture import VAE
import torch
from torch.optim import Adam
from tqdm import tqdm, trange
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from dataset import get_dataloader, get_train_dataset
from img_util import save_side_by_side_image
print("VAE trainer loaded.")
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"

dataloader = get_dataloader(get_train_dataset())


net = VAE()
net.to(device)
net.train()

LEARNING_RATE = 0.001

optimizer = Adam(net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

EPOCHS = 50
PATH = "checkpoint.pt"
writer = None
for i in trange(EPOCHS):
    pbar = tqdm(dataloader)
    current_loss = 0
    most_recent_run_imgs = None
    last_img_batch = None
    running_sum = 0
    idx = 0
    for image_batch in pbar:
        last_img_batch = image_batch.to("cpu").detach().clone().numpy() * 255 
        last_img_batch = last_img_batch.astype(np.uint8)
        optimizer.zero_grad()
        desc = f"Loss: {round(current_loss,4)}"
        
        pbar.set_description(desc+" | prep")

        
        image_batch = image_batch.to(device)
        pbar.set_description(desc+" | eval")

        result = net(image_batch)

        most_recent_run_imgs = result.to("cpu").detach().clone().numpy()* 255 
        most_recent_run_imgs = most_recent_run_imgs.astype(np.uint8) 
        pbar.set_description(desc+" | loss")
        loss = criterion(result, image_batch).to(device)
        current_loss = loss.item()
        running_sum += current_loss
        pbar.set_description(desc+" | back")
        loss.backward()
        pbar.set_description(desc+" | step")
        optimizer.step()


        if idx % 50 == 0:
            
            print("\nSaving checkpoint\n")
            save_side_by_side_image(np.transpose(most_recent_run_imgs[0],(1,2,0)),np.transpose(last_img_batch[0],(1,2,0)), f"train_imgs/{i}_{idx}_generated.png")
            
            torch.save(net.state_dict(), f"ckpt/epoch_{i}_{PATH}")
        idx += 1
    if not writer:
        writer = SummaryWriter()

    writer.add_scalar("Loss/train", running_sum/(len(dataloader)), i) 

    os.remove(f"train_imgs/{i}_generated.png")
    save_side_by_side_image(np.transpose(most_recent_run_imgs[0],(1,2,0)),np.transpose(last_img_batch[0],(1,2,0)))
    torch.save(net.state_dict(), f"ckpt/epoch_{i}_{PATH}")