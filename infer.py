from run_net import run_ddpm
from architecture import Unetv2
import numpy as np
import torch
from bert_vectorize import vectorize_text_with_bert
from PIL import Image


device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"

with torch.no_grad():
    net = Unetv2()
    net.to(device)
    net.eval()
    result = run_ddpm(net, torch.Tensor(vectorize_text_with_bert("A dog")), device=device)
    result = result.to("cpu").detach().clone().numpy()* 255 
    result = result.astype(np.uint8) 
    Image.fromarray(np.transpose(result[0],(1,2,0))).save(f"output.png")
