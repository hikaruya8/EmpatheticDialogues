from transformers import BertForSequenceClassification
from torchsummary import summary
import torch
import numpy as np
import random
from argparse import ArgumentParser

from model_BERT_finetuning import make_model
from dataloader_BERT_finetuning import get_utterance_and_context_loader

import ipdb

parser = ArgumentParser()
parser.add_argument('-n', '--num_epochs', default=3, type=int)
parser.add_argument('-b', '--batch_size', default=32, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_dl, val_dl, num_epochs, batch_size):
    dataloaders_dict = {"train":train_dl, "val": val_dl}
    model, optimizer = make_model(model)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = torch.nn.DataParallel(model)
    model.to(device)
    torch.backends.cudnn.benchmark = True #GPUが効率良く使える

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            batch_processed_num = 0

            # データローダーからミニバッチを取り出す
            for batch in (dataloaders_dict[phase]):
                inputs = batch.utterance[0].to(device)
                labels = batch.label.to(device)

                # optimizerの初期化
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                   # 5. BERTモデルでの予測とlossの計算、backpropの実行
                    outputs = model(inputs, token_type_ids=None, attention_mask=None, labels=labels)
                    # loss and accuracy
                    loss, logits = outputs[:2]
                    _, preds = torch.max(logits, 1)

                    if phase =='train':
                        loss.backward()
                        optimizer.step()

                    curr_loss = loss.item() * inputs.size(0)
                    epoch_loss += curr_loss
                    curr_corrects = (torch.sum(preds==labels.data)).to('cpu').numpy() / inputs.size(0)
                    epoch_corrects += torch.sum(preds==labels.data)

                batch_processed_num += 1
                if batch_processed_num % 10 == 0 and batch_processed_num != 0:
                    print('Processed : ', batch_processed_num * batch_size, ' Loss : ', curr_loss, ' Accuracy : ', curr_corrects)

            # loss and corrects per epoch
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} | Loss:{:.4f} Acc:{:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss, epoch_acc))

    return model


def eval_model(model_trained, test_dl):
    model_trained.to(device)
    epoch_corrects = 0

    for batch in (test_dl):
        inputs = batch.utterance[0].to(device)
        labels = batch.label.to(device)

        with torch.set_grad_enabled(False):
            # input to BertForSequenceClassifier
            outputs = model_trained(inputs, token_type_ids=None, attention_mask=None, labels=labels)
            # loss and accuracy
            loss, logits = outputs[:2]
            _, preds = torch.max(logits, 1)
            epoch_corrects += torch.sum(preds == labels.data)

    epoch_acc = epoch_corrects.double() / len(test_dl.dataset)
    print('Correct rate {} records : {:.4f}'.format(len(test_dl.dataset), epoch_acc))

    # save model
    torch.save(model_trained.state_dict(), 'weights/bert_finetuned_trainded.pth')

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    train_dl, val_dl, test_dl = get_utterance_and_context_loader()
    pre_trained_weights = 'bert-base-uncased'
    model = BertForSequenceClassification.from_pretrained(pre_trained_weights, num_labels=2)
    model.to(device)
    model_trained = train_model(model, train_dl, val_dl, args.num_epochs, args.batch_size)
    eval_model(model_trained, test_dl)

