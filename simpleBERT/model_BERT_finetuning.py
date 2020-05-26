from transformers import BertForSequenceClassification
from torchsummary import summary
import torch
import numpy as np
import random
import torch.optim as optim

from dataloader_BERT_finetuning import get_utterance_and_context_loader
import ipdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_model(model):
    '''# BERTの1〜11段目は更新せず、12段目とSequenceClassificationのLayerのみトレーニングする。'''

    # 一旦すべてのパラメータのrequires_gradをFalseで更新
    for name, param in model.named_parameters():
        param.requires_grad = False

    # BERT encoderの最終レイヤのrequires_gradをTrueで更新
    for name, param in model.bert.encoder.layer[-1].named_parameters():
        param.requires_grad = True

    # 最後のclassificationレイヤのrequires_gradをTrueで更新
    for name, param in model.classifier.named_parameters():
        param.requires_grad = True

    # 4. Optimizerの設定
    optimizer = optim.Adam([
    {'params': model.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
    {'params': model.classifier.parameters(), 'lr': 5e-5}], betas=(0.9, 0.999))

    return model, optimizer

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    pre_trained_weights = 'bert-base-uncased'
    model = BertForSequenceClassification.from_pretrained(pre_trained_weights, num_labels=2)
    model.to(device)
    make_model(model)
