from ddpm import run_ddpm
import torch
from dataset import get_train_dataset, get_dataloader
from bert_vectorize import vectorize_text_with_bert
from torch.optim import Adam
from architecture import Unetv2
import torch.nn as nn
from ddpm import run_ddpm
from tqdm import tqdm, trange
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import os

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"

dataloader = get_dataloader(get_train_dataset(), batch_size=1) # memory


net = Unetv2()
net.to(device)
net.train()

LEARNING_RATE = 0.01

optimizer = Adam(net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

EPOCHS = 50
PATH = "checkpoint.pt"
writer = None
for i in trange(EPOCHS):
    pbar = tqdm(dataloader)
    current_loss = 0
    most_recent_run_imgs = None

    running_sum = 0
    idx = 0
    for image_batch, prompt_batch in pbar:
        optimizer.zero_grad()
        desc = f"Loss: {round(current_loss,4)}"
        
        pbar.set_description(desc+" | prep")

        prompt_batch = prompt_batch.to(device)
        image_batch = image_batch.to(device)
        pbar.set_description(desc+" | eval")
        result = run_ddpm(net, prompt_batch, image_batch, device=device)
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
            print("EIEIEIE")
            os.remove(f"train_imgs/{i}_generated.png")
            Image.fromarray(np.transpose(most_recent_run_imgs[0],(1,2,0))).save(f"train_imgs/{i}_generated.png")
            torch.save(net.state_dict(), f"ckpt/epoch_{i}_{PATH}")
        idx += 1
    if not writer:
        writer = SummaryWriter()

    writer.add_scalar("Loss/train", running_sum/(len(dataloader)), i) 

    os.remove(f"train_imgs/{i}_generated.png")
    Image.fromarray(np.transpose(most_recent_run_imgs[0],(1,2,0))).save(f"train_imgs/{i}_generated.png")
    torch.save(net.state_dict(), f"ckpt/epoch_{i}_{PATH}")

    

