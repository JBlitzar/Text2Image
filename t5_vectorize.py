from transformers import T5Tokenizer, T5Model
import torch


# https://huggingface.co/docs/transformers/en/model_doc/t5
"""
T5 comes in different sizes:

google-t5/t5-small

google-t5/t5-base

google-t5/t5-large

google-t5/t5-3b

google-t5/t5-11b."""

# Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-large')
model = T5Model.from_pretrained('google-t5/t5-large', output_hidden_states=True)
model.eval()

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

def vectorize_text_with_t5(text):
    # Tokenize and get input tensors
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Get hidden states from the encoder
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    
    hidden_states = outputs.last_hidden_state
    
    # Mean pooling to get a single vector representation of the text
    text_representation = torch.mean(hidden_states, dim=1).squeeze(0)
    
    return text_representation

import gc
def cleanup():
    global model
    global tokenizer
    del model
    del tokenizer
    gc.collect()

if __name__ == "__main__":
    text = "A man walking down the street with a dog holding a balloon in one hand."
    text_representation = vectorize_text_with_t5(text)

    print("Vectorized representation:", text_representation)
    print(text_representation.shape)
