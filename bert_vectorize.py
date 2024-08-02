from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
import torch
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
model.eval()

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = model.to(device)
def vectorize_text_with_bert(text):# from hf docs
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    last_layer_hidden_states = hidden_states[-1]
    text_representation = torch.mean(last_layer_hidden_states, dim=1).squeeze(0)

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
    text_representation = vectorize_text_with_bert(text)


    print("Vectorized representation:", text_representation)
    print(text_representation.shape)
