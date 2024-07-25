from factories import UNet_conditional
import re
from dataset import get_train_dataset, get_dataloader, get_random_test_data
import torch
from tqdm import tqdm, trange
from logger import log_data, init_logger, log_img, save_grid_with_label
import torchvision
from wrapper import DiffusionManager, Schedule
from torcheval.metrics import FrechetInceptionDistance
import os
os.system(f"caffeinate -is -w {os.getpid()} &")

RESUME = 0


IS_TEMP = False
if IS_TEMP:
    print("Note: istemp set to true!")




EXPERIMENT_DIRECTORY = "runs/run_4_jxa_fid"






if not IS_TEMP and RESUME == 0:

    os.mkdir(EXPERIMENT_DIRECTORY)

    os.mkdir(EXPERIMENT_DIRECTORY+"/ckpt")

    os.mkdir(EXPERIMENT_DIRECTORY+"/train_img")



device = "mps" if torch.backends.mps.is_available() else "cpu"

dataloader = get_dataloader(get_train_dataset(), batch_size=16)

metric = FrechetInceptionDistance(device=device)
epoch_step_metric = FrechetInceptionDistance(device=device)


net = UNet_conditional(num_classes=768)

if RESUME > 0:
    net.load_state_dict(torch.load(f"{EXPERIMENT_DIRECTORY}/ckpt/latest.pt"))


net.to(device)

wrapper = DiffusionManager(net, device=device, noise_steps=1000)
wrapper.set_schedule(Schedule.LINEAR)

EPOCHS = 50
if IS_TEMP:
    EPOCHS = 5
learning_rate = 3e-4

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


init_logger(dir=EXPERIMENT_DIRECTORY+"/tensorboard")


def generate_sample_save_images(path):
   

    path = os.path.join(EXPERIMENT_DIRECTORY, "train_img", path)
    _, rand_label, rand_label_string = get_random_test_data()
    del _


    generated = wrapper.sample(64, rand_label).detach().cpu()

    row_size = tuple(generated.shape)[0]

    generated_remapped = (generated - generated.min()) / (generated.max() - generated.min())

    generated = torch.concat((generated, generated_remapped))
    save_grid_with_label(torchvision.utils.make_grid(generated, nrow=row_size),rand_label_string, path)

    return generated
    


    

for epoch in trange(EPOCHS, dynamic_ncols=True):

    if epoch < RESUME:
        continue
    last_batch = None

    running_total = 0
    num_runs = 0



    metric.reset()
    for step, (batch, label, _) in enumerate(pbar := tqdm(dataloader, dynamic_ncols=True)):
        epoch_step_metric.reset()

        optimizer.zero_grad()

        loss = wrapper.training_loop_iteration(optimizer, batch, label, criterion)

        running_total += loss
        
        num_runs += 1


        #optimizer.step()
        last_batch = batch[0].detach().cpu()


        pbar.set_description(f"Loss: {'%.4f' % loss}")
        if step % 500 == 499:
            generated = generate_sample_save_images(f"epoch_{epoch}_step_{step}.png")
            metric.update(generated.clip(0,1), False)
            metric.update(batch[:generated.size(0)].clip(0,1), True)
            epoch_step_metric.update(generated.clip(0,1), False)
            epoch_step_metric.update(batch[:generated.size(0)].clip(0,1), True)


            log_data({
                "Loss/Step/Train":loss
            },epoch * len(dataloader) + step)

            log_data({
                "FID/Step/Train":epoch_step_metric.compute().item()
            },epoch * len(dataloader) + step)

            if not IS_TEMP:
                with open(f"{EXPERIMENT_DIRECTORY}/ckpt/latest.pt", "wb+") as f:
                    torch.save(net.state_dict(),f)


    
    
    
    

    tqdm.write(f"Loss: {running_total/num_runs}")

    log_data({
        "Loss/Train":running_total/num_runs
        },epoch)
    log_data({
                "FID/Train":metric.compute().item()
    },epoch)
    
    if not IS_TEMP:
        with open(f"{EXPERIMENT_DIRECTORY}/ckpt/latest.pt", "wb+") as f:
            torch.save(net.state_dict(),f)
    
    if epoch % 1 == 0:
       generate_sample_save_images(f"epoch_{epoch}.png")

    if epoch % 10 == 0 :
        with open(f"{EXPERIMENT_DIRECTORY}/ckpt/epoch_{epoch}.pt", "wb+") as f:
            torch.save(net.state_dict(),f)