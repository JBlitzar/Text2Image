from factories import UNet_conditional_efficient
import re
from dataset import get_train_dataset, get_dataloader, get_random_test_data
import torch
from tqdm import tqdm, trange
from logger import log_data, init_logger, log_img, save_grid_with_label
import torchvision
from wrapper import DiffusionManager, Schedule, ImplicitDiffusionManager
from torcheval.metrics import FrechetInceptionDistance
import os
os.system(f"caffeinate -is -w {os.getpid()} &")

RESUME = 0


IS_TEMP = False
if IS_TEMP:
    print("Note: istemp set to true!")




EXPERIMENT_DIRECTORY = "runs/run_8_large_t5_efficient_batch"

ACCUMULATION_STEPS = 8






if not IS_TEMP and RESUME == 0:
    try:

        os.mkdir(EXPERIMENT_DIRECTORY)

        os.mkdir(EXPERIMENT_DIRECTORY+"/ckpt")

        os.mkdir(EXPERIMENT_DIRECTORY+"/train_img")
    except FileExistsError as e:
        print(f"FileExistsError caught ({EXPERIMENT_DIRECTORY})")
        file_count = sum([len(files) for root, dirs, files in os.walk(EXPERIMENT_DIRECTORY)])
        print(f"File count: {file_count}")
        if file_count <= 1:
            pass
        else:
            print("Exiting.")
            exit()




device = "mps" if torch.backends.mps.is_available() else "cpu"

dataloader = get_dataloader(get_train_dataset(), batch_size=16)

metric = FrechetInceptionDistance(device="cpu") # NotImplementedError: The operator 'aten::_linalg_eigvals' is not currently implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
epoch_step_metric = FrechetInceptionDistance(device="cpu")


net = UNet_conditional_efficient(num_classes=1024)
print(net)
# net.load_state_dict(torch.load(f"runs/run_3_jxa/ckpt/latest.pt"))
if RESUME > 0:
    net.load_state_dict(torch.load(f"{EXPERIMENT_DIRECTORY}/ckpt/latest.pt"))


net.to(device)


wrapper = DiffusionManager(net, device=device, noise_steps=1000) # ImplicitDiffusionManager(net, device=device, noise_steps=1000)
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
    rand_data, rand_label, rand_label_string = get_random_test_data()



    generated = wrapper.sample(64, rand_label).detach().cpu()

    row_size = tuple(generated.shape)[0]

    generated_remapped = (generated - generated.min()) / (generated.max() - generated.min())

    generated = torch.concat((generated, generated_remapped))
    save_grid_with_label(torchvision.utils.make_grid(generated, nrow=row_size),rand_label_string, path)

    return generated, rand_data

def generate_imgs_for_fid():

    rand_data, rand_label, rand_label_string = get_random_test_data(16)



    generated = wrapper.sample_multicond(64, rand_label).detach().cpu()



    return generated.clip(0,1).to("cpu"), rand_data.clip(0,1).to("cpu")


    

for epoch in trange(EPOCHS, dynamic_ncols=True):

    if epoch < RESUME:
        continue
    last_batch = None

    running_total = 0
    num_runs = 0



    metric.reset()
    optimizer.zero_grad()
    for step, (batch, label, _) in enumerate(pbar := tqdm(dataloader, dynamic_ncols=True)):
        epoch_step_metric.reset()


        loss = wrapper.training_loop_iteration(batch, label, criterion)

        running_total += loss
        
        num_runs += 1


        last_batch = batch[0].detach().cpu()


        pbar.set_description(f"Loss: {'%.4f' % loss}")
        
        if step % ACCUMULATION_STEPS == ACCUMULATION_STEPS - 1:
            optimizer.step()
            optimizer.zero_grad()



        if step % 1000 == 999:
            generate_sample_save_images(f"epoch_{epoch}_step_{step}.jpg")
            generated, data = generate_imgs_for_fid()
            print(generated.shape)
            print(data.shape)
            metric.update(generated.clip(0,1).to("cpu"), False)
            metric.update(data.clip(0,1).to("cpu"), True)
            epoch_step_metric.update(generated.clip(0,1).to("cpu"), False)
            epoch_step_metric.update(data.clip(0,1).to("cpu"), True)


            log_data({
                "Loss/Step/Train":loss
            },epoch * len(dataloader) + step)

            log_data({
                "FID/Step/Train":epoch_step_metric.compute().item()
            },epoch * len(dataloader) + step)

            if not IS_TEMP:
                with open(f"{EXPERIMENT_DIRECTORY}/ckpt/latest.pt", "wb+") as f:
                    torch.save(net.state_dict(),f)


    # # just in case
    optimizer.step()
    optimizer.zero_grad()
    
    

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