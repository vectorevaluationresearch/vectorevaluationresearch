import torch
import torch.nn as nn

from transformers import RobertaModel


class BugDetectionModel(nn.Module):
    def __init__(self):
        super(BugDetectionModel, self).__init__()
        self.num_labels = 2
        self.roberta = RobertaModel.from_pretrained('microsoft/codebert-base')

        for param in self.roberta.parameters():
            param.requires_grad = True

        self.dense = nn.Linear(768, 768)  # < Hidden size, Hidden size >
        self.dropout = nn.Dropout(0.1)  # Classifier dropout
        self.out_proj = nn.Linear(768, self.num_labels)

    def forward(self, input_ids, attention_masks, labels):
        model_output = self.roberta(input_ids,
                                    attention_mask=attention_masks,
                                    output_hidden_states=True)

        hidden_states = model_output.hidden_states
        # hidden_states has four dimensions: the layer number (for e.g., 13),
        # batch number (for e.g., 8), token number (for e.g., 256),
        # hidden units (for e.g., 768).

        # Choice: First Layer / Last layer / Concatenation of last four layers /
        # Sum of all layers.
        tok_states = hidden_states[-1]
        cls_output = tok_states[:, 0]
        # print(cls_output.shape)
        pooler = self.dense(cls_output)

        return pooler, tok_states


class Fcs(nn.Module):
    def __init__(self):
        super(Fcs, self).__init__()
        self.num_labels = 2

        self.dense = nn.Linear(768, 768)  # < Hidden size, Hidden size >
        self.dense2 = nn.Linear(768 * 2, 768)
        self.dropout = nn.Dropout(0.1)  # Classifier dropout
        self.out_proj = nn.Linear(768, self.num_labels)

    def forward(self, input_):
        pooler = self.dense2(input_)
        # print(pooler.shape)
        pooler = self.dense(pooler)
        # print(pooler.shape)
        pooler = torch.nn.ReLU()(pooler)
        # print(pooler.shape)
        pooler = self.dropout(pooler)
        # print(pooler.shape)
        cls_pooler = self.out_proj(pooler)
        # print(cls_pooler.shape)

        return cls_pooler

