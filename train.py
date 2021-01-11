from dataloader import EuropData, create_train_test_val
from model import BertTextInfilling
from helper_functions import save_model
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import math


def train(data_path, max_length, num_epochs, learning_rate, batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # load model
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # train params
    training_params = {"batch_size": batch_size,
                       "shuffle": True,
                       "num_workers": 0}

    test_params = {"batch_size": batch_size,
                       "shuffle": True,
                       "num_workers": 0}

    print("==========Loading Data===========")

    train, validate, _ = create_train_test_val(data_path)
    train_set = EuropData(train, max_length, tokenizer)
    val_set = EuropData(validate, max_length, tokenizer)

    train_loader = DataLoader(train_set, **training_params)
    val_loader = DataLoader(val_set, **test_params)

    print("==========Data Loaded===========")

    # params
    num_training_steps = len(train_loader) * num_epochs  # batches into the number of training epochs

    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    train_loss, val_loss = [], []
    best_perplexity = 100

    for epoch in range(num_epochs):

        model.train()
        total_train_loss = 0
        total_val_loss = 0

        print("------------Start Training------------")

        print("Epoch {} of {}".format(epoch+1, num_epochs))

        for iter, batch in enumerate(train_loader):

            batch = (t.type(torch.LongTensor).to(device) for t in batch)
            inputs, attn, labels = batch

            optimizer.zero_grad()
            output = model(inputs.squeeze(dim=1), attn.squeeze(dim=1), labels=labels.squeeze(dim=1))
            loss, logits = output[:2]
            total_train_loss += loss.item()

            print("Train Batch: {} --- Train Loss: {}".format(iter+1, loss))

            loss.backward()

            # Clip the norm of the gradients to 1
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        mean_loss = total_train_loss/len(train_loader)
        train_loss.append(mean_loss)

        print("------------Validate Model------------")
        model.eval()

        for iter, batch in enumerate(val_loader):

            batch = (t.type(torch.LongTensor).to(device) for t in batch)
            inputs, attn, labels = batch

            with torch.no_grad():
                output = model(inputs.squeeze(dim=1), attn.squeeze(dim=1), labels=labels.squeeze(dim=1))
                loss = output[0]
                total_val_loss += loss.item()

                print("Validation Batch: {} --- Validation Loss: {}".format(iter + 1, loss))

        mean_loss = total_val_loss / len(val_loader)
        perplexity = math.exp(mean_loss)
        print("Perplexity: {}".format(perplexity))
        val_loss.append(mean_loss)

        if perplexity < best_perplexity:
            best_perplexity = perplexity
            print('----------Saving model-----------')
            save_model(model=model, model_name='test_bert.pt', tokenizer=tokenizer)
            print('----------Model saved-----------')

    print('----------Training Complete-----------')


if __name__ == "__main__":
    train(data_path='data/europarl-v7.fr-en.en', max_length=128, num_epochs=3, learning_rate=2e-5, batch_size=32)
