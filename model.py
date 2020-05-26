import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from argparse import ArgumentParser
from empchat.transformer_local import TransformerAdapter
from empchat.models import load as load_model
import ipdb

# only the last layer parameters are updated, the others remain fixed.
# feature_extract = True

parser = ArgumentParser()
parser.add_argument('-pm', '--pretrained_model_path', default='./raw_model/transformer.pth', type=str)
parser.add_argument('-ps', '--pretrained_state_dict_path', default='normal_transformer_finetuned.mdl', type=str)
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
print(args)

# modelをロード
pretrained_model = torch.load(args.pretrained_model_path, map_location=args.device)
# 学習済みのパラメータをロード
pretrained_state_dict = torch.load(args.pretrained_state_dict_path, map_location=args.device)

