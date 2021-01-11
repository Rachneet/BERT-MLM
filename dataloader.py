from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from helper_functions import mask_tokens, group_texts
# torch.set_default_tensor_type(torch.FloatTensor)
from transformers import BertTokenizer
from tqdm import tqdm


class EuropData(Dataset):

    def __init__(self, data, max_length, tokenizer):

        self.max_length = max_length
        self.tokenizer = tokenizer
        # self.data = group_texts(data)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        # encoded = tokenizer(self.data[item])
        # print(len(encoded['input_ids']))
        # Encode the sentence
        encoded = self.tokenizer.encode_plus(
            text=self.data[item],  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=self.max_length,  # maximum length of a sentence
            padding='max_length',  # Add [PAD]s
            return_attention_mask=True,  # Generate the attention mask
            return_tensors='pt',  # ask the function to return PyTorch tensors
            truncation=True,
            return_special_tokens_mask=True
        )
        # print(encoded['input_ids'].size())

        inputs, labels = mask_tokens(encoded['input_ids'], tokenizer=self.tokenizer)
        # print(inputs, labels)

        return inputs, labels, encoded['attention_mask']


def get_masked_input_and_labels(text, tokenizer):

    encoded_texts = text.to(torch.float32)

    # 15% BERT masking
    inp_mask = torch.rand(*encoded_texts.shape) < 0.15
    # Do not mask special tokens
    inp_mask[encoded_texts <= 102] = False
    # Set targets to -100 by default, it means ignore
    labels = -100 * torch.ones(encoded_texts.shape)
    # Set labels for masked tokens
    labels[inp_mask] = encoded_texts[inp_mask]

    # Prepare input
    encoded_texts_masked = torch.clone(encoded_texts)
    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = inp_mask & (torch.rand(*encoded_texts.shape) < 0.90)
    encoded_texts_masked[inp_mask_2mask] = tokenizer.mask_token_id  # mask token is the last in the dict

    # Set 10% to a random token
    inp_mask_2random = inp_mask_2mask & (torch.rand(*encoded_texts.shape) < 1 / 9)

    encoded_texts_masked[inp_mask_2random] = torch.randint(
        tokenizer.mask_token_id + 1, len(tokenizer.get_vocab()) + 1, inp_mask_2random.sum().size()
    ).to(torch.float32)

    # y_labels would be same as encoded_texts i.e input tokens
    y_labels = labels.to(torch.float32)

    return encoded_texts_masked, y_labels


def create_train_test_val(path):
    data = open(path, encoding='utf-8').read().strip().split('\n')[:10000]
    train, validate, test = np.split(np.array(data), [int(len(data) * 0.8), int(len(data) * 0.9)])
    return train, validate, test


# test your functions
if __name__ == "__main__":
    # create_train_test_val('data/europarl-v7.fr-en.en')

    # train params
    # training_params = {"batch_size": 32,
    #                    "shuffle": False,
    #                    "num_workers": 0}
    #
    # test_params = {"batch_size": 32,
    #                "shuffle": False,
    #                "num_workers": 0}
    #
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # train, validate, _ = create_train_test_val('data/europarl-v7.fr-en.en')
    # train_set = EuropData(train, 128, tokenizer)
    # val_set = EuropData(validate, 128, tokenizer)
    #
    # train_loader = DataLoader(train_set, **training_params)
    # val_loader = DataLoader(val_set, **test_params)
    #
    # for iter, batch in enumerate(train_loader):
    #     inp, label, attn = batch
    #     # print(inp.shape, label.shape, attn.shape)
    #     break

    data = open('data/europarl-v7.fr-en.en', encoding='utf-8').read().strip().split('\n')[:1000]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded = [tokenizer(text) for text in data]
    # grouper = group_texts(data)
    # print(grouper)