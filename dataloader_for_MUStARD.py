import json
import torchtext
import string
import re
import random
import torchtext
from torchtext.vocab import Vectors

def preprocessing_text(text):
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

        text = text.replace(".", " . ")
        text = text.replace(",", " , ")

        return text

# 分かち書き
def tokenizer_punctuation(text):
    return text.strip().split()

# 前処理と分かち書きをまとめる
def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_punctuation(text)
    return ret


def get_utterance_and_context_loader(max_length=256, batch_size=64):
    max_length = max_length
    batch_size = batch_size

    ID = torchtext.data.Field(sequential=False, use_vocab=False)
    UTTERANCE = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
    SPEAKER = torchtext.data.Field(sequential=False, use_vocab=True)
    CONTEXT_ALL = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False, preprocessing=lambda l: 0 if l == 'True' else 1, is_target=True)
   #  CONTEXT1 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT2 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT3 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT4 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT5 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT6 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT7 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT8 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT9 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   #  CONTEXT10 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
   # CONTEXT11 =  torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")


    train_val_ds = torchtext.data.TabularDataset(
        path='./MUStARD/sarcasm_data.csv', format='csv',
        fields=[("id", ID),
                ("utterance", UTTERANCE),
                ("speaker", SPEAKER),
                ("context_all", CONTEXT_ALL)],
                skip_header=True)

    # test dataloader
    print(f'データ数{len(train_val_ds)}')
    print(f'1つ目のデータ{vars(train_val_ds[1])}')


if __name__ == '__main__':
    get_utterance_and_context_loader()




