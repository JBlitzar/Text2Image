from ddpm import run_ddpm
import torch
from dataset import get_train_dataset, get_dataloader
from bert_vectorize import vectorize_text_with_bert
from torch.optim import Adam
from architecture import Unet
import torch.nn as nn
from ddpm import run_ddpm
from tqdm import tqdm, trange
import numpy as np


device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"

dataloader = get_dataloader(get_train_dataset())


net = Unet()
net.to(device)
net.train()

LEARNING_RATE = 0.01

optimizer = Adam(net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

EPOCHS = 5

for i in trange(EPOCHS):
    pbar = tqdm(dataloader)
    current_loss = 0
    for image_batch, prompt_batch in pbar:
        desc = f"Loss: {round(current_loss,4)}"
        pbar.set_description(desc+" | prep | ")

        #TODO: remove ugly list casting, pre-cache vectorized text
        averaged_vectorizd_prompts = []
        for captions in prompt_batch:
            captions_vectorized = torch.Tensor([list(vectorize_text_with_bert(caption)) for caption in captions])
            new_item = torch.sum(captions_vectorized, 0)/len(captions)
            averaged_vectorizd_prompts.append(new_item)
            
        averaged_vectorizd_prompts = torch.cat(averaged_vectorizd_prompts, dim=0)

        averaged_vectorizd_prompts = averaged_vectorizd_prompts.to(device)
        image_batch = image_batch.to(device)
        pbar.set_description(desc+" | eval | ")
        result = run_ddpm(net, averaged_vectorizd_prompts, image_batch, device=device)
        pbar.set_description(desc+" | loss | ")
        loss = criterion(result, image_batch).to(device)
        pbar.set_description(desc+" | back | ")
        loss.backward()
        pbar.set_description(desc+" | step | ")
        optimizer.step()


