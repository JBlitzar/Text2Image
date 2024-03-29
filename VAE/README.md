# VAE

Essentially a mini-project to make a VAE in order to support the diffusion model in the parent directory
Converts the 3x256x256 image into a 8x8x8, then back into a 3x256x256 image
Both halves of the model are accessible just by doing VAE.encoder and VAE.decoder
Sequential architecture
dataset.py is re-used, because the new dataset has some tweaks, like only including images and not prompts.
