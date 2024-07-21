# Notes.md

## Documentation on this project

Thursday Jul 18 afternoon: 3h coding session

- All right, so I've copied over some code from the Diffusion repository, rewored bert-vectorize, and hit the go button! This is going to take a while (~2h epochs looks like), but the loss is looking _very_ promising! 0.05 after 100 steps epoch 1 (which is really good! Almost wondering if there's something fishy going on). Anyways, in terms of implementation, we'll see how this goes, unclear if I am going to need to do some crossattention or if my boy bert will be able to do most of the heavy lifting. Exciting project is actually getting started! Unet architecture is from `dome272/Diffusion-Models-Pytorch` (I have a fork at `JBlitzar/Diffusion-Models-Pytorch` but _my_ actual diffusion project is at `JBlitzar/Diffusion` [repo might be privated]) because that was the only version I was able to get to work with the Diffusion project. 78652bd

- Lets see how this trains! :) 7a8ceaa

- Added stuff so that it samples with label, and outputs a pretty image with the label on top. More bug-squashing, the cycle continues. 1bf05c6

- Trained for 500 steps and loss was looking good, image range was for some reason -34 to 27 or something?? Added sigmoid, and loss went way up. So I'm going to remove it. Checking input data it is securly between 0 and 1, so I dont know whats going on. This happened in the diffusion project, so I'm just going to trust the process. But I added a little thing where it shows a sampled image unremapped and remapped. I promise I wont interrupt training this time. 1e43aaa

Jul 19 training session:

- Wow! Training is looking pretty good! 45c89b3
- Main bottleneck is distilbert vectorization: taking a long time to evaluate: Model was on cpu. So I moved it to mps. Also added dynamic_ncols to the progress bar. 38bb336
- Seems like I'm able to get away without crossattention because of the self attention + bert. Results are good, but not photorealistic so I'll see if that will be an issue. So far, loss hasn't plateaued, so we will continue training!
- Btw after training for a bit remapped and clipped look basically the same
- In terms of image quality, you definitely can tell some features, like a prompt with "house" looks like a house, but you defenitely couldn't tell what it was just given the image. some images are solid white
- improvements: crossattention, cosine schedule (+ less timesteps), deeper conditional (prompt) encoding rather than appending it to the timestep?
-

Jul 19 coding session afternoon:

- Okay, I've paused the current training to implement some improvements. I added crossattention, which took quite a bit of debugging. I switched over to cosine schedule and 500 noising steps. I also increased batch size to 32 from 16. Lets hit the go button and see what happens!
- Seems right off the bat that the loss is a bit higher. So I increased learning rate by a factor of 3 since I increased the model complexity. (ig? It seemed to quell). Training is a bit slower, similar results of ~0.05 loss at step 400
- I have confidence in the crossattention + selfattention pair, I think it will be great!
- Things to tweak: lr, batch size, x/sa ordering, model architecture size, schedule start/end/timesteps/type, etc
- I think I will train the original model overnight so that we can see if it gets better, otherwise start in on this new one. (which I will need to fix up cuz theres problems with shapes)

Jul 20

- yeah, so the original architecture plateaued at loss=0.01 and stayed there, results seem to look basically all the same.
- Started up the attention version, fixed the problem when sampling by repeating the condition tensor (I cant believe we werent doing that before and it would work!)
- Again, promising loss: 0.05 after 100 steps

Jul 21:

- Ok, so it (`run_2_xa_cos`) converged to 0.03 after 7 epochs, so its safe to say that model performance was worse. Might be owing to the 500 noise steps, since I realized openai did got away with 50 noise steps _at inference_.
- I'm going to try with just adding the cross-attention. I also switched it up so that C is injected just @ the crossattention, not also added to `t`.
- This is taking a while, I'm going to try a just selfattn run first.
