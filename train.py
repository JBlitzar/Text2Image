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
for i in trange(EPOCHS):
    pbar = tqdm(dataloader)
    current_loss = 0
    most_recent_run_imgs = None


    for image_batch, prompt_batch, idx in pbar:
        optimizer.zero_grad()
        desc = f"Loss: {round(current_loss,4)}"
        pbar.set_description(desc+" | prep")

        prompt_batch = prompt_batch.to(device)
        image_batch = image_batch.to(device)
        pbar.set_description(desc+" | eval")
        result = run_ddpm(net, prompt_batch, image_batch, device=device)
        most_recent_run_imgs = result.to("cpu").detach().clone().numpy().astype(np.uint8)
        pbar.set_description(desc+" | loss")
        loss = criterion(result, image_batch).to(device)
        pbar.set_description(desc+" | back")
        loss.backward()
        pbar.set_description(desc+" | step")
        optimizer.step()
        
        if idx % 50 == 0:
            Image.fromarray(np.transpose(most_recent_run_imgs[0],(1,2,0))).save(f"train_imgs/{i}_generated.png")
            torch.save(net.state_dict(), f"ckpt/epoch_{i}_{PATH}")
            
    
    Image.fromarray(np.transpose(most_recent_run_imgs[0],(1,2,0))).save(f"train_imgs/{i}_generated.png")
    torch.save(net.state_dict(), f"ckpt/epoch_{i}_{PATH}")

    

