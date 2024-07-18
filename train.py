from factories import UNet_conditional

from dataset import get_train_dataset, get_dataloader
import torch
from tqdm import tqdm, trange
from logger import log_data, init_logger, log_img
import torchvision
from wrapper import DiffusionManager, Schedule

import os
os.system(f"caffeinate -is -w {os.getpid()} &")


RESUME = 0


IS_TEMP = False
if IS_TEMP:
    print("Note: istemp set to true!")




EXPERIMENT_DIRECTORY = "runs/run_1_coco64_domecond"






if not IS_TEMP and RESUME == 0:

    os.mkdir(EXPERIMENT_DIRECTORY)

    os.mkdir(EXPERIMENT_DIRECTORY+"/ckpt")



device = "mps" if torch.backends.mps.is_available() else "cpu"

dataloader = get_dataloader(get_train_dataset(), batch_size=16)



net = UNet_conditional(num_classes=256)

if RESUME > 0:
    net.load_state_dict(torch.load(f"{EXPERIMENT_DIRECTORY}/ckpt/latest.pt"))


net.to(device)

wrapper = DiffusionManager(net, device=device)
wrapper.set_schedule(Schedule.LINEAR)

EPOCHS = 100
if IS_TEMP:
    EPOCHS = 5
learning_rate = 3e-4

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


init_logger(dir=EXPERIMENT_DIRECTORY+"/tensorboard")
#init_logger(net, next(iter(dataloader))[0].to(device), dir=EXPERIMENT_DIRECTORY+"/tensorboard")
for epoch in trange(EPOCHS):

    if epoch < RESUME:
        continue
    last_batch = None
    last_generated = None
    running_total = 0
    num_runs = 0




    for step, (batch, label) in enumerate(pbar := tqdm(dataloader)):

        optimizer.zero_grad()

        loss = wrapper.training_loop_iteration(optimizer, batch, label, criterion)

        running_total += loss
        
        num_runs += 1


        #optimizer.step()
        last_batch = batch[0].detach().cpu()


        pbar.set_description(f"Loss: {'%.2f' % loss}")
        if step % 500 == 499:
            last_generated = wrapper.sample(64).detach().cpu()
            log_img(torchvision.utils.make_grid(last_generated),f"train_img/epoch_{epoch}_step_{step}.png")


    
    
    
    

    tqdm.write(f"Loss: {running_total/num_runs}")

    log_data({
        "Loss/Train":running_total/num_runs
        },epoch)
    
    if not IS_TEMP:
        with open(f"{EXPERIMENT_DIRECTORY}/ckpt/latest.pt", "wb+") as f:
            torch.save(net.state_dict(),f)
    
    if epoch % 1 == 0:
        last_generated = wrapper.sample(64).detach().cpu()
        log_img(torchvision.utils.make_grid(last_generated),f"train_img/epoch_{epoch}.png")
    if epoch % 10 == 0 :
        with open(f"{EXPERIMENT_DIRECTORY}/ckpt/epoch_{epoch}.pt", "wb+") as f:
            torch.save(net.state_dict(),f)