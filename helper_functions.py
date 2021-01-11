import torch
import os
from transformers import BertForMaskedLM, BertTokenizer
from typing import Tuple, Optional


def save_model(model, model_name, tokenizer):
    output_dir = './model_save/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    # torch.save(model_to_save.state_dict(), output_dir+model_name)
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_vocabulary(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))


def load_model(path, model_class):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model_class
    # model.load_state_dict(torch.load(path, map_location="cuda:0"))  # Choose whatever GPU device number you want

    # state_dict = torch.load(path, map_location="cuda:0")
    # model = BertForMaskedLM.from_pretrained(path, from_pt=True)
    model= BertForMaskedLM.from_pretrained(path)
    model.to(device)
    return model


# def mask_tokens(inputs: torch.Tensor, mask_labels: torch.Tensor, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
#     'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
#     """
#
#     if tokenizer.mask_token is None:
#         raise ValueError(
#             "This tokenizer does not have a mask token which is necessary for masked language modeling."
#             " Remove the --mlm flag if you want to use this tokenizer."
#         )
#     labels = inputs.clone()
#     # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability
#     # defaults to 0.15 in Bert/RoBERTa)
#
#     probability_matrix = mask_labels
#
#     special_tokens_mask = [
#         tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
#     ]
#     probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
#     if tokenizer._pad_token is not None:
#         padding_mask = labels.eq(tokenizer.pad_token_id)
#         probability_matrix.masked_fill_(padding_mask, value=0.0)
#
#     masked_indices = probability_matrix.bool()
#     labels[~masked_indices] = -100  # We only compute loss on masked tokens
#
#     # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
#     indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#     inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
#
#     # 10% of the time, we replace masked input tokens with random word
#     indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#     random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
#     inputs[indices_random] = random_words[indices_random]
#
#     # The rest of the time (10% of the time) we keep the masked input tokens unchanged
#     return inputs, labels

def mask_tokens(
        inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None, tokenizer=None,
        mlm_probability=0.15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def group_texts(examples, block_size=128):
    # Concatenate all texts.
    concatenated_examples = ' '.join(map(str, examples))

    # print(concatenated_examples)
    total_length = len(concatenated_examples)
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size

    # Split by chunks of max_len.
    text_split = concatenated_examples.split()
    result = [
         ' '.join(text_split[i:i+block_size]) for i in range(0, len(text_split), block_size)
    ]
    # print(len(result[0]))
    # print(result[0])
    # result["labels"] = result["input_ids"].copy()
    return result