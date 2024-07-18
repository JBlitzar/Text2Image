# Notes.md

## Documentation on this project

Jul 18: 3h coding session

- All right, so I've copied over some code from the Diffusion repository, rewored bert-vectorize, and hit the go button! This is going to take a while (~2h epochs looks like), but the loss is looking _very_ promising! 0.05 after 100 steps epoch 1 (which is really good! Almost wondering if there's something fishy going on). Anyways, in terms of implementation, we'll see how this goes, unclear if I am going to need to do some crossattention or if my boy bert will be able to do most of the heavy lifting. Exciting project is actually getting started! Unet architecture is from `dome272/Diffusion-Models-Pytorch` (I have a fork at `JBlitzar/Diffusion-Models-Pytorch` but _my_ actual diffusion project is at `JBlitzar/Diffusion` [repo might be privated]) because that was the only version I was able to get to work with the Diffusion project. 78652bd

- Lets see how this trains! :) 7a8ceaa

- Added stuff so that it samples with label, and outputs a pretty image with the label on top. More bug-squashing, the cycle continues. 1bf05c6

- Trained for 500 steps and loss was looking good, image range was for some reason -34 to 27 or something?? Added sigmoid, and loss went way up. So I'm going to remove it. Checking input data it is securly between 0 and 1, so I dont know whats going on. This happened in the diffusion project, so I'm just going to trust the process. But I added a little thing where it shows a sampled image unremapped and remapped. I promise I wont interrupt training this time. 1e43aaa
