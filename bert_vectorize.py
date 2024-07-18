from transformers import BertTokenizer, BertModel
import torch
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
model = BertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
model.eval()
def vectorize_text_with_bert(text):# from hf docs
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    last_layer_hidden_states = hidden_states[-1]
    text_representation = torch.mean(last_layer_hidden_states, dim=1).squeeze(0)

    return text_representation

if __name__ == "__main__":
    text = "A man walking down the street with a dog holding a balloon in one hand."
    text_representation = vectorize_text_with_bert(text)


    print("Vectorized representation:", text_representation)
    print(len(text_representation))
