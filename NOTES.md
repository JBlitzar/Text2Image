# Notes.md

## Documentation on this project

Jul 18

- All right, so I've copied over some code from the Diffusion repository, rewored bert-vectorize, and hit the go button! This is going to take a while (~2h epochs looks like), but the loss is looking _very_ promising! 0.05 after 100 steps epoch 1. Anyways, in terms of implementation, we'll see how this goes, unclear if I am going to need to do some crossattention or if my boy bert will be able to do most of the heavy lifting. Exciting project is actually getting started! Unet architecture is from `dome272/Diffusion-Models-Pytorch` (I have a fork at `JBlitzar/Diffusion-Models-Pytorch` but _my_ actual diffusion project is at `JBlitzar/Diffusion` [repo might be privated]) because that was the only version I was able to get to work with the Diffusion project. 78652bd

- Lets see how this trains! :) 7a8ceaa

- Added stuff so that it samples with label, and outputs a pretty image with the label on top. More bug-squashing, the cycle continues. 1bf05c6
