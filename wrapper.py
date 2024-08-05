import torch
import torch.nn as nn
from enum import Enum
from tqdm import trange





Schedule = Enum('Schedule', ['LINEAR', 'COSINE'])

class DiffusionManager(nn.Module):
    def __init__(self, model: nn.Module, noise_steps=1000, start=0.0001, end=0.02, device="cpu", **kwargs ) -> None:
        super().__init__(**kwargs)

        self.model = model

        self.noise_steps = noise_steps

        self.start = start
        self.end = end
        self.device = device

        self.schedule = None

        self.set_schedule()

        #model.set_parent(self)


    def _get_schedule(self, schedule_type: Schedule = Schedule.LINEAR):
        if schedule_type == Schedule.LINEAR:
            return torch.linspace(self.start, self.end, self.noise_steps)
        elif schedule_type == Schedule.COSINE:
            # https://arxiv.org/pdf/2102.09672 page 4
            #https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py 
            #line 18
            def get_alphahat_at(t):
                def f(t):
                    s=self.start
                    return torch.cos((t/self.noise_steps + s)/(1+s) * torch.pi/2) ** 2
                
                return f(t)/f(torch.zeros_like(t))

            t = torch.Tensor(range(self.noise_steps))

            t = 1-(get_alphahat_at(t + 1)/get_alphahat_at(t))
            
            t = torch.minimum(t, torch.ones_like(t) * 0.999) #"In practice, we clip β_t to be no larger than 0.999 to prevent singularities at the end of the diffusion process n"

            return t
    
    def set_schedule(self, schedule: Schedule = Schedule.LINEAR):
        self.schedule = self._get_schedule(schedule).to(self.device)
    
    def get_schedule_at(self, step):
        beta = self.schedule
        alpha = 1 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)

        return self._unsqueezify(beta.data[step]), self._unsqueezify(alpha.data[step]), self._unsqueezify(alpha_hat.data[step])
    
    @staticmethod
    def _unsqueezify(value):
        return value.view(-1, 1, 1, 1)#.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
    def noise_image(self, image, step):

        
        image = image.to(self.device)

        beta, alpha, alpha_hat = self.get_schedule_at(step)

        epsilon = torch.randn_like(image)

        # print(alpha_hat)

        # print(alpha_hat.size())
        # print(image.size())

        noised_img = torch.sqrt(alpha_hat) * image  + torch.sqrt(1 - alpha_hat) * epsilon

        return noised_img, epsilon
    
    def random_timesteps(self, amt=1):

        return torch.randint(low=1, high=self.noise_steps, size=(amt,))
    
    

    
    def sample(self, img_size, condition, amt=5, use_tqdm=True):

        if tuple(condition.shape)[0] < amt:
            condition = condition.repeat(amt, 1)

        self.model.eval()

        condition = condition.to(self.device)

        my_trange = lambda x, y, z: trange(x,y, z, leave=False,dynamic_ncols=True)
        fn = my_trange if use_tqdm else range
        with torch.no_grad():
            
            cur_img = torch.randn((amt, 3, img_size, img_size)).to(self.device)
            for i in fn(self.noise_steps-1, 0, -1):

                timestep = torch.ones(amt) * (i)

                timestep = timestep.to(self.device)



                predicted_noise = self.model(cur_img, timestep, condition)

                beta, alpha, alpha_hat = self.get_schedule_at(i)

                cur_img = (1/torch.sqrt(alpha))*(cur_img - (beta/torch.sqrt(1-alpha_hat))*predicted_noise)
                if i > 1:
                    cur_img = cur_img + torch.sqrt(beta)*torch.randn_like(cur_img)


        self.model.train()




    
        return cur_img
    def sample_multicond(self, img_size, condition, use_tqdm=True):
        num_conditions = condition.shape[0]

        
        
        amt = num_conditions

        self.model.eval()

        condition = condition.to(self.device)

        my_trange = lambda x, y, z: trange(x, y, z, leave=False, dynamic_ncols=True)
        fn = my_trange if use_tqdm else range
        
        with torch.no_grad():

            cur_img = torch.randn((amt, 3, img_size, img_size)).to(self.device)
            
            for i in fn(self.noise_steps-1, 0, -1):
                timestep = torch.ones(amt) * i
                timestep = timestep.to(self.device)


                predicted_noise = self.model(cur_img, timestep, condition)

                beta, alpha, alpha_hat = self.get_schedule_at(i)

                cur_img = (1 / torch.sqrt(alpha)) * (cur_img - (beta / torch.sqrt(1 - alpha_hat)) * predicted_noise)
                if i > 1:
                    cur_img = cur_img + torch.sqrt(beta) * torch.randn_like(cur_img)

        self.model.train()

        # Return images sampled for each condition
        return cur_img
    
    def training_loop_iteration(self, optimizer, batch, label, criterion):

        def print_(string):
            for i in range(10):
                print(string)



        batch = batch.to(self.device)

        #label = label.long() # uncomment for nn.Embedding
        label = label.to(self.device)

        timesteps = self.random_timesteps(batch.shape[0]).to(self.device)

        noisy_batch, real_noise = self.noise_image(batch, timesteps)

        if torch.isnan(noisy_batch).any() or torch.isnan(real_noise).any():
            print_("NaNs detected in the noisy batch or real noise")


        pred_noise = self.model(noisy_batch, timesteps, label)

        if torch.isnan(pred_noise).any():
            print_("NaNs detected in the predicted noise")

        loss = criterion(real_noise, pred_noise)

        if torch.isnan(loss).any():
            print_("NaNs detected in the loss")

        loss.backward()


        return loss.item()
    



class ImplicitDiffusionManager(DiffusionManager):
    
    def __init__(self, model: nn.Module, noise_steps=1000, start=0.0001, end=0.02, device="cpu", **kwargs) -> None:
        super().__init__(model, noise_steps, start, end, device, **kwargs)


    def sample(self, img_size, condition, amt=5, use_tqdm=True, sampling_steps=100, sigma=0): # Sigma represents stochasticity. Zero = deterministic
        if tuple(condition.shape)[0] < amt:
            condition = condition.repeat(amt, 1)

        self.model.eval()
        condition = condition.to(self.device)
        my_trange = lambda x, y, z: trange(x, y, z, leave=False, dynamic_ncols=True)
        fn = my_trange if use_tqdm else range

        with torch.no_grad():
            cur_img = torch.randn((amt, 3, img_size, img_size)).to(self.device)
            for i in fn(sampling_steps-1, 0, -1):
                timestep = torch.ones(amt) * i
                timestep = timestep.to(self.device)
                predicted_image = self.model(cur_img, timestep, condition)
                if i > 0:
                    _, _, alpha_hat_prev = self.get_schedule_at(i-1)
                    cur_img = torch.sqrt(alpha_hat_prev) * predicted_image + torch.sqrt(1 - alpha_hat_prev - sigma**2) * torch.randn_like(cur_img)
                else:
                    cur_img = predicted_image
        self.model.train()
        return cur_img

    def sample_multicond(self, img_size, condition, use_tqdm=True, sampling_steps=100, sigma=0):
        num_conditions = condition.shape[0]
        amt = num_conditions
        self.model.eval()
        condition = condition.to(self.device)
        my_trange = lambda x, y, z: trange(x, y, z, leave=False, dynamic_ncols=True)
        fn = my_trange if use_tqdm else range

        with torch.no_grad():
            cur_img = torch.randn((amt, 3, img_size, img_size)).to(self.device)
            for i in fn(sampling_steps-1, 0, -1):
                timestep = torch.ones(amt) * i
                timestep = timestep.to(self.device)
                predicted_image = self.model(cur_img, timestep, condition)
                if i > 0:
                    _, _, alpha_hat_prev = self.get_schedule_at(i-1)
                    cur_img = torch.sqrt(alpha_hat_prev) * predicted_image + torch.sqrt(1 - alpha_hat_prev - sigma **2) * torch.randn_like(cur_img)
                else:
                    cur_img = predicted_image
        self.model.train()
        return cur_img

    def training_loop_iteration(self, batch, label, criterion):
        def print_(string):
            for i in range(10):
                print(string)
        
        batch = batch.to(self.device)
        label = label.to(self.device)
        timesteps = self.random_timesteps(batch.shape[0]).to(self.device)
        noisy_batch, real_noise = self.noise_image(batch, timesteps)

        if torch.isnan(noisy_batch).any() or torch.isnan(real_noise).any():
            print_("NaNs detected in the noisy batch or real noise")

        pred_image = self.model(noisy_batch, timesteps, label)

        if torch.isnan(pred_image).any():
            print_("NaNs detected in the predicted image")

        loss = criterion(batch, pred_image)

        if torch.isnan(loss).any():
            print_("NaNs detected in the loss")

        loss.backward()
        return loss.item()