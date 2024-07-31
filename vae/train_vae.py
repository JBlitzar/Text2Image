from architecture import VAE_loss, COCO_CVAE_factory, COCO_CVAE_Shallow_factory
from dataset import get_train_dataset, get_test_dataset, get_dataloader    
import torch
from tqdm import tqdm, trange
from logger import log_data, init_logger, log_img
import torchvision


import os
os.system(f"caffeinate -is -w {os.getpid()} &")


EXPERIMENT_DIRECTORY = "runs/run4_shallowog"

os.mkdir(EXPERIMENT_DIRECTORY)

os.mkdir(EXPERIMENT_DIRECTORY+"/ckpt")


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device {device}")


dataloader = get_dataloader(get_train_dataset())


testloader = get_dataloader(get_test_dataset())


KL_WEIGHT = 0.05#1#0.002

net, _ = COCO_CVAE_Shallow_factory(device=device, num_classes=64, start_depth=32)
net.to(device)
EPOCHS = 400
learning_rate = 1e-3
criterion = VAE_loss
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


print(next(iter(dataloader))[0].to(device).size())
#init_logger(net, next(iter(dataloader))[0][0].to(device).unsqueeze(0), dir=EXPERIMENT_DIRECTORY+"/tensorboard")
init_logger(net, dir=EXPERIMENT_DIRECTORY+"/tensorboard")
for epoch in trange(EPOCHS):
    last_batch = None
    last_generated = None
    running_total = 0
    num_runs = 0
    running_total_kl = 0
    running_total_reconstruction = 0



    for batch, labels, _ in (pbar := tqdm(dataloader)):
        del _ #humanreadable labels unused
        optimizer.zero_grad()

        batch = batch.to(device)
        labels = labels.to(device)



        results, mean, logvar = net(batch, labels)



        loss, reconstruction, kl = criterion(batch, results, mean, logvar, KL_WEIGHT)

        pbar.set_description(f"loss {'%.4f' % loss.item()} |r {'%.4f' % reconstruction.item()} |kl {'%.4f' % kl.item()}")

        loss.backward()

        running_total += loss.item()
        running_total_kl += kl.item()
        running_total_reconstruction += reconstruction.item()
        
        num_runs += 1


        optimizer.step()
        last_batch = batch[0].detach().cpu()
        last_generated = results[0].detach().cpu()
    

    num_test_runs = 0
    running_total_test = 0
    running_total_kl_test = 0
    running_total_reconstruction_test = 0
    with torch.no_grad():
        if testloader != None:
            for batch, labels, _ in tqdm(testloader, leave=False):
                del _ #unused

                batch = batch.to(device)

                labels = labels.to(device)

                results, mean, logvar = net(batch, labels)

                loss, reconstruction, kl = criterion(batch, results, mean, logvar, KL_WEIGHT)

                running_total_test += loss.item()
                running_total_kl_test += kl.item()
                running_total_reconstruction_test += reconstruction.item()
                num_test_runs += 1
            
            log_data({
                "Loss/Test":running_total_test/num_test_runs,
                "Loss/Test/KL": running_total_kl_test/num_test_runs,
                "Loss/Test/Reconstruction": running_total_reconstruction_test / num_test_runs
            }, epoch)
    print()
    print(f"Loss: {running_total/num_runs}")

    log_data({
        "Loss/Train":running_total/num_runs,
        "Loss/KL":running_total_kl/num_runs, 
        "Loss/Reconstruction":running_total_reconstruction/num_runs,
        
        },epoch)

    with open(f"{EXPERIMENT_DIRECTORY}/ckpt/latest.pt", "wb+") as f:
        torch.save(net.state_dict(),f)
    if epoch % 1 == 0 :
        log_img(torchvision.utils.make_grid([last_batch, last_generated]),f"train_img/epoch_{epoch}.png")
        #save_side_by_side_image(last_batch, last_generated, f"train_img/epoch_{epoch}.png")
    if epoch % 10 == 0:
        with open(f"{EXPERIMENT_DIRECTORY}/ckpt/epoch_{epoch}.pt", "wb+") as f:
            torch.save(net.state_dict(),f)
