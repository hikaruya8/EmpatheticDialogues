from transformers import BertForSequenceClassification
from torchsummary import summary
import torch
from torchtext import data
import sys
import numpy as np
import random
import logging
import datetime
from argparse import ArgumentParser

from model_BERT_finetuning import make_model
from dataloader_kfold_BERT_finetuning import LoadData

import ipdb

parser = ArgumentParser()
parser.add_argument('-n', '--num_epochs', default=3, type=int)
parser.add_argument('-b', '--batch_size', default=32, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
# Loading Arguments
if len(sys.argv) <= 1:
    raise Exception('Please give json settings file path!')
args_p = Path(sys.argv[1])
if args_p.exists() is False:
    raise Exception('Path not found. Please check an argument again!')

with args_p.open(mode='r') as f:
    true = True
    false = False
    null = None
    args = json.load(f)

# logging
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
logfile = str('log/log-{}.txt'.format(run_start_time))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(logfile),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)
'''

pre_trained_weights = 'bert-base-uncased'

def main(model, optimizer, num_epochs, batch_size):
    # logger.info("***** Running Training *****")
    # logger.info(f"Now fold: {fold_index + 1} / {args['num_folds']}")

    data_generator = LoadData()
    _history = [] #k-foldのためリストで保持しておく
    fold_index = 0


    for SENTENCE, LABEL, train_data, val_data in data_generator.get_fold_data(num_folds=5):
        # logger.info("***** Running Training *****")
        # logger.info(f"Now fold: {fold_index + 1} / {args['num_folds']}")
        # dataloaders_dict = {"train":train_dl, "val": val_dl}
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True #GPUが効率良く使える

        train_dl = data.Iterator(train_data, batch_size=batch_size, sort_key=lambda x: len(x.utterance), device=device)
        val_dl = data.Iterator(val_data, batch_size=batch_size, sort_key=lambda x: len(x.utterance), device=device)

        # dataloaders_dict = {"train":train_dl, "val": val_dl}

        for epoch in range(num_epochs):
            train_loss, train_acc = train_run(model, train_dl, optimizer)
            print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        val_loss, val_acc = eval_run(model, val_dl)
        print(f'Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc:.2f}% |')
        _history.append([val_loss, val_acc])
        fold_index += 1

    _history = np.asarray(_history)
    loss = np.mean(_history[:, 0])
    acc = np.mean([h.item() for h in _history[:, 1]])

    print('***** Cross Validation Result *****')
    print(f'LOSS: {loss}, ACC: {acc}')


def train_run(model, iterator, optimizer):
    epoch_loss = 0.0
    epoch_corrects = 0.0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        inputs = batch.utterance[0].to(device)
        labels = batch.label.to(device)
        outputs = model(inputs, token_type_ids=None, attention_mask=None, labels=labels)
        # output, _ = model(batch.utterance)
        # loss = criterion(output, batch.label)
        loss, logits = outputs[:2]
        _, preds = torch.max(logits, 1)
        # acc = binary_accuracy(output, batch.label)
        loss.backward()
        optimizer.step()
        # epoch_loss += loss.item()
        # epoch_acc += acc.item()
        curr_loss = loss.item() * inputs.size(0)
        epoch_loss += curr_loss
        curr_corrects = (torch.sum(preds==labels.data)).to('cpu').numpy() / inputs.size(0)
        epoch_corrects += torch.sum(preds==labels.data)

    return epoch_loss / len(iterator), epoch_corrects / len(iterator)


def eval_run(model, iterator):
    epoch_loss = 0.0
    epoch_corrects = 0.0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            inputs = batch.utterance[0].to(device)
            labels = batch.label.to(device)
             # input to BertForSequenceClassifier
            outputs = model(inputs, token_type_ids=None, attention_mask=None, labels=labels)
            # loss and accuracy
            loss, logits = outputs[:2]
            _, preds = torch.max(logits, 1)
            curr_loss = loss.item() * inputs.size(0)
            epoch_loss += curr_loss
            curr_corrects = (torch.sum(preds==labels.data)).to('cpu').numpy() / inputs.size(0)
            epoch_corrects += torch.sum(preds==labels.data)

    return epoch_loss / len(iterator), epoch_corrects / len(iterator)

    epoch_acc = epoch_corrects.double() / len(test_dl.dataset)
    print('Correct rate {} records : {:.4f}'.format(len(test_dl.dataset), epoch_acc))

    # save model
    torch.save(model_trained.state_dict(), 'weights/bert_kfold_finetuned_trainded.pth')


    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # train_dl, val_dl, test_dl = get_utterance_and_context_loader()
    model = BertForSequenceClassification.from_pretrained(pre_trained_weights, num_labels=2)
    model.to(device)
    model, optimizer = make_model(model)
    model.to(device)
    main(model, optimizer, args.num_epochs, args.batch_size)




#         for epoch in range(num_epochs):
#             for phase in ['train', 'val']:
#                 if phase == 'train':
#                     model.train()
#                 else:
#                     model.eval()

#                 epoch_loss = 0.0
#                 epoch_corrects = 0
#                 batch_processed_num = 0

#                 # データローダーからミニバッチを取り出す
#                 for batch in (dataloaders_dict[phase]):
#                     inputs = batch.utterance[0].to(device)
#                     labels = batch.label.to(device)

#                     # optimizerの初期化
#                     optimizer.zero_grad()

#                     with torch.set_grad_enabled(phase=='train'):
#                        # BERTモデルでの予測とlossの計算、backpropの実行
#                         outputs = model(inputs, token_type_ids=None, attention_mask=None, labels=labels)
#                         # loss and accuracy
#                         loss, logits = outputs[:2]
#                         _, preds = torch.max(logits, 1)

#                         if phase =='train':
#                             loss.backward()
#                             optimizer.step()

#                         curr_loss = loss.item() * inputs.size(0)
#                         epoch_loss += curr_loss
#                         curr_corrects = (torch.sum(preds==labels.data)).to('cpu').numpy() / inputs.size(0)
#                         epoch_corrects += torch.sum(preds==labels.data)

#                         if phase == 'train':
#                             kf_train_history.append([epoch_loss, epoch_acc])
#                             train_fold_index += 1
#                         if phase == 'val':
#                             kf_val_history.append(epoch_loss, epoch_acc)
#                             val_fold_index += 1

#                     batch_processed_num += 1
#                     if batch_processed_num % 10 == 0 and batch_processed_num != 0:
#                         print('Processed : ', batch_processed_num * batch_size, ' Loss : ', curr_loss, ' Accuracy : ', curr_corrects)

#                 # loss and corrects per epoch
#                 epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
#                 epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

#                 print('Epoch {}/{} | {:^5} | Loss:{:.4f} Acc:{:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss, epoch_acc))


#                 kf_val_history = np.asarray(kf_val_history)

#         # logger.info('***** Cross Validation Result *****')
#         # logger.info(f'LOSS: {loss}, ACC: {acc}')

#         return model


# def eval_model(model_trained, test_dl):
#     model_trained.to(device)
#     epoch_corrects = 0

#     for batch in (test_dl):
#         inputs = batch.utterance[0].to(device)
#         labels = batch.label.to(device)

#         with torch.set_grad_enabled(False):
#             # input to BertForSequenceClassifier
#             outputs = model_trained(inputs, token_type_ids=None, attention_mask=None, labels=labels)
#             # loss and accuracy
#             loss, logits = outputs[:2]
#             _, preds = torch.max(logits, 1)
#             epoch_corrects += torch.sum(preds == labels.data)

#     epoch_acc = epoch_corrects.double() / len(test_dl.dataset)
#     print('Correct rate {} records : {:.4f}'.format(len(test_dl.dataset), epoch_acc))

#     # save model
#     torch.save(model_trained.state_dict(), 'weights/bert_finetuned_trainded.pth')

# if __name__ == '__main__':

#     model_trained = train_model(model, train_dl, val_dl, args.num_epochs, args.batch_size)
#     eval_model(model_trained, test_dl)

