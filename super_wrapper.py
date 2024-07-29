from torch.nn.modules import Module
from wrapper import DiffusionManager
import torch
from tqdm import trange
import torch.nn as nn


class SuperResDiffusionManager(DiffusionManager):
    def __init__(self, model: Module, pretrainedGenWrapper: DiffusionManager, noise_steps=1000, start=0.0001, end=0.02, device="cpu", **kwargs) -> None:
        super().__init__(model, noise_steps, start, end, device, **kwargs)

        self.pretrainedGen = pretrainedGenWrapper
        self.pretrainedGen.model.eval()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def sample(self, img_size, condition, amt=5, use_tqdm=True):

        self.model.eval()

        cur_img = self.pretrainedGen.sample(img_size, condition, amt, use_tqdm)

        if tuple(condition.shape)[0] < amt:
            condition = condition.repeat(amt, 1)

        self.model.eval()

        condition = condition.to(self.device)

        my_trange = lambda x, y, z: trange(x,y, z, leave=False,dynamic_ncols=True)
        fn = my_trange if use_tqdm else range
        with torch.no_grad():
            
            #cur_img = torch.randn((amt, 3, img_size, img_size)).to(self.device)
            for i in fn(self.noise_steps-1, 0, -1):

                timestep = torch.ones(amt) * (i)

                timestep = timestep.to(self.device)



                predicted_noise = self.model(cur_img, timestep, condition)

                beta, alpha, alpha_hat = self.get_schedule_at(i)

                cur_img = (1/torch.sqrt(alpha))*(cur_img - (beta/torch.sqrt(1-alpha_hat))*predicted_noise)
                if i > 1:
                    cur_img = cur_img + torch.sqrt(beta)*torch.randn_like(cur_img)
                    cur_img = self.pool(cur_img)





        self.model.train()
        self.pretrainedGen.model.eval()
    
        return cur_img
    def sample_multicond(self, img_size, condition, use_tqdm=True):
        

        self.model.eval()

        

        cur_img = self.pretrainedGen.sample_multicond(img_size, condition, use_tqdm)


        num_conditions = condition.shape[0]

        amt = num_conditions

        self.model.eval()

        condition = condition.to(self.device)

        my_trange = lambda x, y, z: trange(x, y, z, leave=False, dynamic_ncols=True)
        fn = my_trange if use_tqdm else range
        
        with torch.no_grad():

            
            
            for i in fn(self.noise_steps-1, 0, -1):
                timestep = torch.ones(amt) * i
                timestep = timestep.to(self.device)


                predicted_noise = self.model(cur_img, timestep, condition)

                beta, alpha, alpha_hat = self.get_schedule_at(i)

                cur_img = (1 / torch.sqrt(alpha)) * (cur_img - (beta / torch.sqrt(1 - alpha_hat)) * predicted_noise)
                if i > 1:
                    cur_img = cur_img + torch.sqrt(beta) * torch.randn_like(cur_img)

                    cur_img = self.pool(cur_img)
                

        self.model.train()
        self.pretrainedGen.model.eval()

        # Return images sampled for each condition
        return cur_img
    
    def training_loop_iteration(self, optimizer, batch, label, criterion):
        self.pretrainedGen.model.eval()
        for param in self.pretrainedGen.model.parameters():
            param.requires_grad = False

        def print_(string):
            for i in range(10):
                print(string)


        optimizer.zero_grad()

        

        #label = label.long() # uncomment for nn.Embedding
        label = label.to(self.device)
        batch = batch.to(self.device)
        
        timesteps = self.random_timesteps(batch.shape[0]).to(self.device)

        with torch.no_grad():
            small_batch = self.pool(batch)
            
            
            
            noisy_batch, real_noise = self.pretrainedGen.noise_image(small_batch, timesteps)

            if torch.isnan(noisy_batch).any() or torch.isnan(real_noise).any():
                print_("NaNs detected in the noisy batch or real noise")


            pred_noise = self.pretrainedGen.model(noisy_batch, timesteps, label)

            if torch.isnan(pred_noise).any():
                print_("NaNs detected in the predicted noise")
        
        

        pred_large_noise = self.model(pred_noise, timesteps, label)

            

        loss = criterion(batch, pred_large_noise)

        if torch.isnan(loss).any():
            print_("NaNs detected in the loss")

        loss.backward()
        optimizer.step()

        return loss.item()