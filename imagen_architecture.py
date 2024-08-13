# https://arxiv.org/pdf/2205.11487
# page 41


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SeamlessDenseLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(SeamlessDenseLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        

        input_flattened_size = np.prod(input_size)
        output_flattened_size = np.prod(output_size)
        

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_flattened_size, output_flattened_size)
        self.unflatten = nn.Unflatten(1, output_size)

    def forward(self, x):

        x = self.flatten(x)
        

        x = self.fc(x)
        
        x = self.unflatten(x)
        
        return x

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module): 
    def __init__(self, channels, size): 
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 8, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
    

class CrossAttention(nn.Module):
    def __init__(self, channels, size, context_dim):
        super(CrossAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.context_dim = context_dim
        self.mha = nn.MultiheadAttention(channels, 8, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.context_ln = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )


        self.context_proj = nn.Linear(context_dim, channels)

    def forward(self, x, context):
        
        # Reshape and permute x for multi-head attention
        batch_size, channels, height, width = x.size()

        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1,2)
        x_ln = self.ln(x)

        # Expand context to match the sequence length of x
        context = self.context_proj(context)

        context = context.unsqueeze(1).expand(-1, x_ln.size(1), -1)

        context_ln = self.context_ln(context)

        



        # Apply cross-attention
        attention_value, _ = self.mha(x_ln, context_ln, context_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value

        # Reshape and permute back to the original format
        return attention_value.permute(0, 2, 1).view(batch_size, channels, height, width)
    


# also called ResNetBlock
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.SiLU(),


            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
            
            
        )

    def forward(self, x):
        if self.residual:
            return F.silu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

# DBlock
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=1024):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(2), # They use a conv, but this works

            nn.Conv2d(in_channels, in_channels, kernel_size=3,stride=2,padding=1),
            ResNetBlock(in_channels, in_channels, residual=True),
            ResNetBlock(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

# UBlock
class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=1024):
        super().__init__()

        self.inc = in_channels
        self.outc = out_channels

        self.up = nn.ConvTranspose2d(in_channels // 2 , in_channels // 2,kernel_size=3,stride=2,padding=1, output_padding=1)#nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            ResNetBlock(in_channels, in_channels, residual=True),
            ResNetBlock(in_channels, out_channels, in_channels // 2), 
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb




    
class ImagenUnet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=1024, num_classes=None, context_dim=None, device="mps"):
        super().__init__()

        if context_dim is None:
            context_dim = num_classes
        self.device = device
        self.time_dim = time_dim


        start_depth = 128 # Must be 128, see page 43
        xa_amt_depth = 64

        self.inc = ResNetBlock(c_in, start_depth)

        self.down1 = DBlock(start_depth, start_depth * 2)
        self.xa1 = CrossAttention(start_depth * 2, xa_amt_depth // 2, context_dim)

        self.down2 = DBlock(start_depth * 2, start_depth * 4)
        self.xa2 = CrossAttention(start_depth * 4, xa_amt_depth // 4, context_dim)

        self.down3 = DBlock(start_depth * 4, start_depth * 4)
        self.xa3 = CrossAttention(start_depth * 4, xa_amt_depth // 8, context_dim)


        self.bot1 = ResNetBlock(start_depth * 4, start_depth * 8)
        self.bot2 = ResNetBlock(start_depth * 8, start_depth * 8)
        self.bot3 = ResNetBlock(start_depth * 8, start_depth * 4)

        self.up1 = UBlock(start_depth * 8, start_depth * 2)
        self.xa4 = CrossAttention(start_depth * 2, xa_amt_depth // 4, context_dim)

        self.up2 = UBlock(start_depth * 4, start_depth)
        self.xa5 = CrossAttention(start_depth, xa_amt_depth // 2, context_dim)

        self.up3 = UBlock(start_depth * 2, start_depth)
        self.xa6 = CrossAttention(start_depth, xa_amt_depth, context_dim)

        self.outc = nn.Conv2d(start_depth, c_out, kernel_size=1)#nn.Sequential(nn.Conv2d(start_depth, c_out, kernel_size=1),SeamlessDenseLayer((c_out,64,64),(c_out,64,64))) # They have a Dense at the end (BTW this contributes to 2/3 of the params!)


        if num_classes is not None:
            self.label_emb = nn.Linear(num_classes, time_dim)#Embedding(num_classes, time_dim)
            self.num_classes = num_classes
            if context_dim is None:
                context_dim = num_classes

            self.context_dim = context_dim

            self.label_crossattn_emb = nn.Linear(num_classes, context_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:

            attn_y = y[:,:self.num_classes]
            attn_y = self.label_crossattn_emb(attn_y)

            # y = y[:,:self.num_classes]

            # y = self.label_emb(y)


            # t += y

        x1 = self.inc(x)

        x2 = self.down1(x1, t)
        x2 = self.xa1(x2, attn_y)
        #x2 = self.sa1(x2)

        x3 = self.down2(x2, t)
        x3 = self.xa2(x3, attn_y)
        #x3 = self.sa2(x3)

        x4 = self.down3(x3, t)
        x4 = self.xa3(x4, attn_y)
        #x4 = self.sa3(x4)


        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)


        x = self.up1(x4, x3, t)
        x = self.xa4(x,attn_y)
        #x = self.sa4(x)

        x = self.up2(x, x2, t)
        x = self.xa5(x, attn_y)
        #x = self.sa5(x)

        x = self.up3(x, x1, t)
        x = self.xa6(x, attn_y)
        #x = self.sa6(x)

        output = self.outc(x)



        #output = F.sigmoid(x)
        return output


if __name__ == "__main__":
    net = ImagenUnet(num_classes=1024).to("mps")

    def count_parameters(model):
        return torch.tensor([p.numel() for p in model.parameters() if p.requires_grad]).sum().item()
    print(f"Parameters: {count_parameters(net)}")

    minibatch = torch.randn((1,3,64,64)).to("mps")

    o = net(minibatch, torch.randint(low=1, high=1000, size=(1,)).to("mps"), torch.randn((1,1024)).to("mps"))

    print(o.size())

    