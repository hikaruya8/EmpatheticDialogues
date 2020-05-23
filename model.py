import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from argparse import ArgumentParser
import ipdb


parser = ArgumentParser()
parser.add_argument('-p', '--pretrained_model_path', default='normal_transformer_finetuned.mdl', type=str)
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
print(args)

pre_trained_model = torch.load(args.pretrained_model_path, map_location=args.device)

class PreTrainWithCosineSimilarity(nn.Module):

    def __init__(self, input_dim):
        super(PreTrainWithCosineSimilarity, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(in_features=self.input_dim, out_features=1, bias=False)
