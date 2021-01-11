from helper_functions import load_model
from model import BertTextInfilling
from transformers import BertTokenizer, pipeline
import torch
import numpy as np


def predict(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('model_save/', BertTextInfilling())
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('model_save/')

    # Encode the sentence
    encoded = tokenizer.encode_plus(
        text=text,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length=128,  # maximum length of a sentence
        padding='max_length',  # Add [PAD]s
        return_attention_mask=True,  # Generate the attention mask
        return_tensors='pt',  # ask the function to return PyTorch tensors
        truncation=True,
        return_special_tokens_mask=True
    )

    mask_idx = np.where(encoded['input_ids'].numpy()[0] == tokenizer.mask_token_id)
    # inp = torch.unsqueeze(encoded['input_ids'], dim=0).to(device)
    # attn = torch.unsqueeze(encoded['attention_mask'], dim=0).to(device)
    inp = encoded['input_ids'].to(device)
    attn = encoded['attention_mask'].to(device)
    output = model(inp, attn)
    logits = output[0].cpu().detach().numpy()

    # pred_mask_word = np.argmax(logits[:, mask_idx, :], axis=3).squeeze()
    pred_token_idx = logits[:, mask_idx, :].squeeze()
    # pred_token_idx = pred_token_idx[995:]
    pred_mask_words = np.argpartition(pred_token_idx, -5)[-5:]
    # print(pred_mask_words)
    replacements = tokenizer.convert_ids_to_tokens(pred_mask_words)

    return replacements



if __name__ == "__main__":
    text = "A product [MASK] is the marketing copy that explains what a product is and why it’s worth purchasing."
    # text = "We look to the [MASK] also to ensure that there is matched funding for projects."
    # text = "I feel an [MASK] of this type is essential to prevent the sort of administrative unwieldiness which too often acts as a brake on the proper application of new conditions."
    replacements = predict(text)
    print(replacements)

    # nlp = pipeline("fill-mask", model='model_save/bert_europ_10k.pt')
    # print(nlp("HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))