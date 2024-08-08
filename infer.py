from factories import UNet_conditional
from wrapper import DiffusionManager, Schedule
import os
import re
import torch
from bert_vectorize import vectorize_text_with_bert, cleanup
import time
import torchvision
from logger import save_grid_with_label
from clip_score import select_top_n_images



EXPERIMENT_DIRECTORY = "runs/run_3_jxa_resumed"
device = "mps" if torch.backends.mps.is_available() else "cpu"

try:
    os.mkdir(os.path.join(EXPERIMENT_DIRECTORY, "inferred"))
except:
    print("Skipping making directory, directory already exists")

net = UNet_conditional(num_classes=768)
net.to(device)
net.load_state_dict(torch.load(os.path.join(EXPERIMENT_DIRECTORY, "ckpt/latest.pt")))
def count_parameters(model):
    return torch.tensor([p.numel() for p in model.parameters() if p.requires_grad]).sum().item()
print(f"Parameters: {count_parameters(net)}")


wrapper = DiffusionManager(net, device=device, noise_steps=1000)
wrapper.set_schedule(Schedule.LINEAR)


def infer(prompt, amt=1, topn=8):

    path = os.path.join(EXPERIMENT_DIRECTORY, "inferred", re.sub(r'[^a-zA-Z\s]', '', prompt).replace(" ", "_")+str(int(time.time()))+".png")

    vprompt = vectorize_text_with_bert(prompt).unsqueeze(0)

    generated = wrapper.sample(64, vprompt, amt=amt).detach().cpu()

    generated, _ = select_top_n_images(generated, prompt, n=topn)

    save_grid_with_label(torchvision.utils.make_grid(generated),prompt + f"({topn} best of {amt})", path)


def run_jobs():
    n=1
    bestof=1
    print(f"using best {bestof} of {n}")
    processed_tasks = set()
    def read_jobs():
        try:
            with open("inference_jobs.txt", 'r') as file:
                tasks = file.readlines()
            return [task.strip() for task in tasks]
        except FileNotFoundError:
            return []
    
    tasks = read_jobs()
    new_tasks = [task for task in tasks if task not in processed_tasks]
    while new_tasks:
        

        if new_tasks:
            for task in new_tasks:
                infer(task, n,bestof)
                processed_tasks.add(task)
        tasks = read_jobs()
        new_tasks = [task for task in tasks if task not in processed_tasks]

    cleanup()

if __name__ == "__main__":
    #infer(input("Prompt? "), 8)
    run_jobs()