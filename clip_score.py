import torch
from transformers import CLIPProcessor, CLIPModel
import gc
from torchvision.transforms.functional import resize

# https://huggingface.co/docs/transformers/en/model_doc/clip
def select_top_n_images(image_tensors: torch.Tensor, text: str, n: int, model_name='openai/clip-vit-base-patch32') -> torch.Tensor:

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)


    if len(image_tensors.shape) != 4 or image_tensors.shape[1] != 3:
        raise ValueError("Image tensor must be of shape (batch_size, 3, height, width)")
    

    inputs = processor(text=text, return_tensors="pt", padding=True)
    

    device = model.device
    image_tensors = resize(image_tensors.to(device),(224,224)) # because skill issue clip can only deal with 224^2 images
    inputs = {k: v.to(device) for k, v in inputs.items()}
    

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=image_tensors)
        text_features = model.get_text_features(**inputs)
    
    del model
    del processor
    gc.collect()

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    

    similarities = torch.nn.functional.cosine_similarity(image_features, text_features)
    

    _, top_n_indices = similarities.topk(n, largest=True)
    top_n_images = image_tensors[top_n_indices.squeeze()]
    
    return top_n_images, similarities[top_n_indices]
