
from transformers import AutoTokenizer, CLIPTextModelWithProjection
import torch
model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = model.to(device)
def vectorize_text_with_bert(text):# from hf docs
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    

    return outputs.text_embeds



if __name__ == "__main__":
    text = "A man walking down the street with a dog holding a balloon in one hand."
    text_representation = vectorize_text_with_bert(text)


    print("Vectorized representation:", text_representation)
    print(text_representation.size())
