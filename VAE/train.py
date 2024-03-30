from vae_architecture import COCO_VAE_factory,vae_loss_function
from dataset import get_train_dataset, get_dataloader, get_val_dataset
import torch
from tqdm import tqdm, trange
from logger import log_data, init_logger, log_img
import torchvision

import os
os.system(f"caffeinate -is -w {os.getpid()} &")


device = "mps" if torch.backends.mps.is_available() else "cpu"

dataloader = get_dataloader(get_val_dataset(), batch_size=32)#get_dataloader(get_train_dataset())


net = COCO_VAE_factory(device=device)
net.to(device)
net.train()
EPOCHS = 500
learning_rate = 0.001
criterion = vae_loss_function#torch.nn.MSELoss()#torch.nn.BCELoss()#
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


first_data = next(iter(dataloader))[0].to(device).unsqueeze(0)
print(first_data.size())
init_logger(net, first_data, dir="runs/cocovae128")


for epoch in trange(EPOCHS):
    last_batch = None
    last_generated = None
    running_total = 0
    num_runs = 0


    running_total_reconstruction = 0
    running_total_kl = 0

    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        batch = batch.to(device)

        results, mean, logvar = net(batch)

        loss, reconstruction, kl = criterion(batch, results, mean, logvar)

        loss.backward()

        running_total += loss.item()
        running_total_reconstruction += reconstruction.item()
        running_total_kl += kl.item()
        num_runs += 1


        optimizer.step()
        last_batch = batch[0].detach().cpu()
        last_generated = results[0].detach().cpu()
    

    print(f"Loss: {running_total/num_runs}")

    log_data({"Loss/Train":running_total/num_runs,"Loss/Reconstruction":running_total_reconstruction/num_runs, "Loss/KL":running_total_kl/num_runs},epoch)
    
    if epoch % 1 == 0 :
        log_img(torchvision.utils.make_grid([last_batch, last_generated]),f"train_img/epoch_{epoch}.png")
    if epoch % 10 == 0:
        with open(f"ckpt/epoch_{epoch}.pt", "wb+") as f:
            torch.save(net.state_dict(),f)
