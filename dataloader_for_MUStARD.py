import json
import torchtext
import string
import re
import random
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
    json_open = json.open('./MUStARD/sarcasm_data.json')

    max_length = max_length
    batch_size = batch_size



