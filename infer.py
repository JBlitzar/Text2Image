from factories import UNet_conditional
from wrapper import DiffusionManager, Schedule
import os
import re
import torch
from bert_vectorize import vectorize_text_with_bert
import time
import torchvision
from logger import save_grid_with_label



EXPERIMENT_DIRECTORY = "runs/run_3_jxa"
device = "mps" if torch.backends.mps.is_available() else "cpu"

try:
    os.mkdir(os.path.join(EXPERIMENT_DIRECTORY, "inferred"))
except:
    print("Skipping making directory, directory already exists")

net = UNet_conditional(num_classes=768)
net.to(device)
net.load_state_dict(torch.load(os.path.join(EXPERIMENT_DIRECTORY, "ckpt/latest.pt")))



wrapper = DiffusionManager(net, device=device, noise_steps=1000)
wrapper.set_schedule(Schedule.LINEAR)


def generate_sample_save_images(prompt, amt=1):

    path = os.path.join(EXPERIMENT_DIRECTORY, "inferred", re.sub(r'[^a-zA-Z\s]', '', prompt).replace(" ", "_")+str(int(time.time()))+".png")

    vprompt = vectorize_text_with_bert(prompt).unsqueeze(0)

    generated = wrapper.sample(64, vprompt, amt=amt).detach().cpu()


    save_grid_with_label(torchvision.utils.make_grid(generated),prompt, path)

if __name__ == "__main__":
    generate_sample_save_images(input("Prompt? "), 8)
