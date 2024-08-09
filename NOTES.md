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
- This is taking a while, I'm going to try a just xattn run first.
- ok. Just xattn run (`run_3_jxa` `jxa` just cross attention) is looking pretty good! Loss is jumpy, but as low as 0.015 halfway through epoch 1. (Ranges 0.1-3)
- `run_3_jxa` 1 epoch later, and its looking pretty good! It took at least until epoch 3 on `run1` to get not-noise images, but by epoch 1 it happened for us
- ok quite a few hours later, results are looking pretty good! (epoch 7) Like it actually looks like the thing its describing. (kind of). Going to let this run overnight

Jul 22:

- Okay, its the morning ( epoch 14 `run_3_jxa`) and things are looking good! Losses around 0.05, going as low as 0.02. wow! Things are coming along, I'll train this up and then set my sights on the super resolution unet.

Jul 25 ~3hr session:

- Okay, it seems to have plateaued, in terms of loss and image quality. Interestinly, `run3` has converged just about the same as `run1`, but perhaps that shows the shortcomings of MSE as a loss metric. Should I get a FID evaluator up and running? So I've added that now. For reference, SOTA models get an FID of 3 at the best on MSCOCO https://paperswithcode.com/sota/text-to-image-generation-on-coco, although I'll be happy if I get a value <100
- Also, thoughts on super resolution, which I will want eventually. https://arxiv.org/pdf/2405.14822v1 says that you literally hot-wire the diffusion model to have a bigger decoder, but freeze the original components, so its not a diffusion model then running through unet, its a diffusion unet with a bigger decoder. Its cost-effective and has faster training because you freeze most, add a bit, freeze most, add a bit, etc.

- Returning from that tangent, I've gotten FID up and working after squashing a few bugs (clipping generated and batch to [0,1] and moving them to cpu). Keep in mind this is the same architecture as before. I could change to cosine, but it's more computationally expensive and doesn't actually add anything except faster inference.
- One improvement: CLIP rather than distilbert because its meant for image-text pipelines
- ===
- Okay after all that rambling, I've started training with FID and clip. Speeds are pretty much the same, however extra 14-20min per epoch for fid calculation (Kind of yikes: might bump the 500 interval to 1k or something: 14 generated images per epoch is a while, bc thats 14/hr or 1 every 4 mins). We'll see how this goes, and of course I'm going to have to go through about 3 days of training time. RAM use is up, 35gb rather than like 20 (7f1e092)
- I bumped up the step checker to every 1k steps, approx every 12 minutes. The fid slowdown is now only 7 mins, and CLIP doesn't seem to incur any slowdown (1.5 it/s) (but does incur higher memory usage rather than distilbert)
- Looking good: 0.05 after 100 steps, 0.02 after 700
- Non-noise images appearing at epoch 2! Looking good.
- Loss 0.01 after 2 epochs. FID is still through the roof understandably
- Epoch 2 step 6999 looks like something! fid is still high
- Epoch 3 just started, FID just went below 300
- Epoch 3 step 2999 Looking pretty good!
- I'm going to run this overnight

Jul 26:

- FID is down to 166, very good!
- By epoch 8 its starting to look like stuff.
- Run4 seems to have the same loss curve as run3
- (in terms of MSE)
- FID 150 at epoch 10
- Loss 0.003

Jul 27:

- Continued training, made a quick inference script and ran it on `jxa`. Performs relatively well (see `runs/jxa/generated`), performs better on captions in the testset, probably alluding to the captioning style, seems to perform better at certain things (like airplanes, bathrooms, and baseball). Possibly investigate a new dataset?

Jul 28:

- Okay, `xa_clip_fid` is doing ok, but I think its time to pause training. So far, `jxa` is our best, so I'm going to try to code up super-resolution.

Jul 29:

- Coded up, but didn't run super-resolution. Looking back on results and notes, I fear the models are horribly overfit. Like how prompts styled like the testset perform better, and how its only really good at airplanes, baseball, and bathrooms. Perhaps I will revisit coco and try oxfordflowers or CUB for a bit. IDK.

Jul 30:

- Here we are, and it's time to change it up. I think I'm going to go with a vq-vae, which is a diffusion model working in the latent space of a VAE. https://paperswithcode.com/paper/vector-quantized-diffusion-model-for-text-to
- Here's a paper that you can see actually works, so I have confidence in MSCOCO. (best FID was 18)
- So here I am, coding it up (check vae folder). I pasted some stuff in from `JBlitzar/VAE-2`, which gave me a head start, now I'm going to try to get that working out.
- First roadblock is loss going to infinity then nan. Root of the problem seems to be KL loss.
- Well, as it goes, I add print statements and suddenly KL isnt exploding anymore. I'll keep an eye on it.

Jul 31:

- Ok, trained VAE overnight and it's as one would expect, pretty good but blurry. I'm going to experiment with vgg loss and see if that's any better (`vae/runs/run_2_vggloss`)
- Reimplemented vgg loss (`vae/runs/run_3_vggI` (vggImproved)) also set KL weight to 1
- Added in MSE loss to the mix, so now reconstruction is MSE + VGG
- Because VGG loss was looking kind of like randomness. Perhaps tune down KL loss again, because its looking kind of like it's dominating the playing field in terms of losses. Probably not as drastic as 0.002 but like 0.5. Loss is also suspiciously low?? like 0.3 (perhaps cuz of not using BCE) Still repeating noise images, turning KL down to 0.05
- `vae/runs/run3_vggmsekl`
- Changed the VGG layers to be [0,1,2,3] because those correspond to fine-grained details
- Well, I ran that and it was bad.
- Perhaps try a smaller one? 16x16x128
- I am going to try a shallow architecture (`vae/runs/run4_shallowog`)
- Well, I found something https://theadamcolton.github.io/your-vae-sucks.html and it talks about stable diffusion, so maybe I get a good vae from there
- BTW found a LAION dataset rehosted (laion took their stuff down from the public https://huggingface.co/datasets/fantasyfish/laion-art)
- Bruh well the "your vae sucks" article's idea didn't work. Maybe I can just yoink dalle vae?
- https://github.com/openai/DALL-E/blob/master/notebooks/usage.ipynb
- Yeah so check it out! https://colab.research.google.com/drive/19R65mTwtacfpAnwMJmrbJ8YZ9o41atmS#scrollTo=VLtw94v76B6j
- I can totally yoink the dalle vae, and its relatively small (0.5gb pkl sizes)
- _a little bit_ cheaty, but I still have to make the actual model, and its nice that it's pretrained

- Meh I'm just going to train `run3_jxa` to hope to feel better about myself and maybe it'll magically become amazing

Aug 2:

- I have a genius plan for increasing quality for outputs of the text to image model. Generate lots, automated select best of n
- Done (clip_score, infer.py changes)

Aug 3:

- More trainign, its prob overfit because FID went up, but we always have run3jxa (nonresumed). Lets see how far this goes. I could just be wasting cpu time, and I prob am. Bestof is looking good

Aug 4

- Low batch size might be to blame for noisiness
- Funny funny technique: gradient accumulation!
- starting up `run5_xa_acc`, simulating batch size 128 with also actual batch size 32 (batch 32, acc 4 = simulated 128)
- Yeah, so the loss stayed >1 on that one, so instead I'm going to try smaller lr.
- Investigated torch.compile, doesn't work on mps :\
- Anyways going to run `run5_xa_lr`

Aug 5

- Perhaps try DDIM https://arxiv.org/pdf/2010.02502 that predicts the noise instead, then we can also incorporate perceptual loss perhaps
- https://ar5iv.labs.arxiv.org/html/2010.02502, https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html gives us the sampling equation
- Made ImplicitDiffusionManager. Experiment one: no change, just using implicit. Experiment 2 will be using vgg loss a little bit as well
- Well, that was a bust. I'm going to remove it from history, but you can always find it @ ffae316
- Re-coding with https://github.com/Alokia/diffusion-DDIM-pytorch/blob/master/utils/engine.py as ref
- Added option for quadratic schedule (https://arxiv.org/pdf/2006.09011)
- Latent diffusion might actually be better because its meant to represent features well
- Well, ddim run didn't really work.
- Panic that inference didn't work, resolved by commenting out a sigmoid

Aug 8

MENTOR MEETING
Thoughts:

Revisit if discouraged:

- p3 DEbug VGG loss? numbers are clamped weirdly (really small, wrong range, -1-1 vs 0-1 vs 0-255) nonsense gray feels like a scaling issue
- p3 Get vqvae code working, depending on how feasable

Do now in order:

- p2 Start simple, simple problem, t2i is really hard. Hard to have a fast prototyping cycle. Simplest dataset? flickr, coco. MNIST for t2i vibes.
- p2 Bigger dataset, or subset (like CUB).
- (if it arises, research, but long term to think abt) Big problem is large model. Large model means more ram, ? Get access to gpu somehow (colab or lambda or aws or smth). Research a good method, would get better results.
- p1 Larger architecture? 25m might be small
- Read imagen paper? we are basically replicating that workflow
- p0 https://arxiv.org/pdf/2205.11487 4.4 we need a bigger text encoder? Get a really big bert or try to get t5
- p0 "efficient" design means to slap downsample/upsample on either end of the unet because less parameters

ok coding.

-

Aug 9:

- Coded up the efficient design and made it lager by increasing start depth and adding another layer of depth. Switched bert to t5 large.
- Old architecture at v2_architecture, old traininer at v2_trainer.
- The efficient architecture has 1 million less parameters :skull: (380 million vs 379 million)
