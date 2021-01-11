from transformers import BertForMaskedLM, BertConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_model_summary import summary
from dataloader import EuropData

class BertTextInfilling(nn.Module):

    def __init__(self):
        super(BertTextInfilling, self).__init__()
        # config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        #                     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, torchscript=True)
        self.bert_mlm = BertForMaskedLM.from_pretrained('bert-base-uncased', torchscript=True)

    def forward(self, token_id, attn_mask, *args):

        # labels = kwargs.get('labels', None)
        if args:
            labels = args[0].squeeze(dim=1)
        else:
            labels = None

        output = self.bert_mlm(token_id.squeeze(dim=1), attn_mask.squeeze(dim=1), labels=labels)
        return output


if __name__ == "__main__":
    # summary(BertTextInfilling(), torch.ones(1,128).to(torch.int64))

    training_params = {"batch_size": 128,
                       "shuffle": True,
                       "num_workers": 0}

    train_set = EuropData('data/train_europa.txt', 128)
    train_loader = DataLoader(train_set, **training_params)

    model = BertTextInfilling()
    # model.cuda()
    for iter, batch in enumerate(train_loader):
        inp, label, attn = batch
        print(inp.shape, label.shape, attn.shape)
        out = model(inp.type(torch.LongTensor), attn.type(torch.LongTensor), label.type(torch.LongTensor))
        print(out)
        break





