from transformers import (BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling, pipeline)
import torch
import numpy as np
import math
import collections
import urllib
import itertools
from torch.utils.data import DataLoader
import pandas as pd
from dataloader import EuropData,create_train_test_val
from datasets import load_dataset
# from helper_functions import group_texts


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab


def save_vocab():
    # retrieve the bert vocab from aws: Done only once
    url = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
    vocab_file = urllib.request.urlretrieve(url, "data/vocab.txt")
    vocab_path = "data/vocab.txt"
    vocab = load_vocab(vocab_path)
    return vocab


# ----------------------------------------------------------------------------------------

def group_texts(examples, max_seq_length=128):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # load model
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    datasets = load_dataset("text", data_files={"train": 'data/train_full.txt',
                                                "validation": 'data/test_full.txt'})

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
    # print(tokenized_datasets['train'][0]['input_ids'])

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=8
    )

    # print(len(lm_datasets['train'][0]['input_ids']))
    # print(len(lm_datasets['train'][0]['labels']))
    # print(lm_datasets['train'][265]['input_ids'])
    # print(lm_datasets['train'][266]['input_ids'])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    training_args = TrainingArguments(
        output_dir="test-clm",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.save_model()


def predict(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    pipe = pipeline('fill-mask', model='test-clm', tokenizer=tokenizer)
    result = pipe(text)
    # print(result)
    for d in result:
        print(d['token_str'])
        print(d['score'])
        print("------------------")
    return result


def create_train_test_split(path):
    data = open(path, encoding='utf-8').read().strip().split('\n')
    print(len(data))
    total_len = len(data)
    train_percent = 0.8
    train_data = data[:int(0.8*total_len)]
    test_data = data[len(train_data):]
    print(len(train_data), len(test_data))

    with open('data/train_full.txt', 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write("%s\n" % item)

    with open('data/test_full.txt', 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write("%s\n" % item)


if __name__ == "__main__":
    # create_train_test_split('data/europarl-v7.fr-en.en')
    # train()
    # text = "A product [MASK] is the marketing copy that explains what a product is and why it’s worth purchasing."
    text = "We look to the [MASK] also to ensure that there is matched funding for projects."
    predict(text)


