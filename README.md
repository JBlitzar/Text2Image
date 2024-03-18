# Text2Image

- Dataset:

  - Conceptual captions: https://ai.google.com/research/ConceptualCaptions/download (waaaay big, images not included (its links), and some broken links)
  - Flickr30k: https://paperswithcode.com/dataset/flickr30k (might be too small, you need to fill out a form to get the data)
  - ✅Coco captions: https://paperswithcode.com/dataset/coco-captions (just about right size, 25gb, direct dl), 640x480. https://cocodataset.org/#download

- Preprocessing

  - Normalize images, tokenize text
  - Vectorize textual descriptions using gpt, bert, or t5
  - Implemented at ~/Documents/python_programs/huggingface/vectorize_bert.py
  - Is fixed length ✅ (768)
  - Architecture
  - Gan (kinda unstable)
  - Vae (autoencoder) (?)
  - Diffusion (DDPM?) (speed should not be an issue with new computer)
  - Ddpm takes a noise vector (noise image) and iteratively denoises it
  - For text to image, the neural network gets inputted both the image and the embedded prompt.
  - Specification of the nn:
  - U-net architecture: image goes down, image goes up, skip connections BUT with the skip connections, the prompt is injected throughout.
  - See https://en.wikipedia.org/wiki/Diffusion_model#Choice_of_architecture
  - We dont need those weird “t2i adapters”
    ![](https://www.researchgate.net/publication/368572081/figure/fig1/AS:11431281120695577@1676603550379/The-overall-architecture-is-composed-of-two-parts-1-a-pre-trained-stable-diffusion.png)

- Train
  - Get text, tokenize, vectorize (bert), run through model (iteratively, ~20 steps, implement in the forward function), compare generated image with ground truth image
  - All the classic stuff such as printing images every epoch and checkpoints
  - Nuance: print over the course of iterations: for each epoch, for each image, print at it iteration 1,5,10,15,20
  - Hyperparameter tuning
  - Essentially step 4 with many different hyperparameter configurations, make sure to have train/test/val
- Running
  - Get text, vectorize (bert), then run thru model
