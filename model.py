import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from argparse import ArgumentParser
from empchat.transformer_local import TransformerAdapter
from empchat.models import load as load_model
import ipdb

class CosineSimilarityWithPretrain(nn.Module):

    def __init__(self, input_dim):
        super(CosineSimilarityWithPretrain, self).__init__()
        self.input_dim = input_dim
        # ここにfinetuning前のmodelを書く
        self.embeddings = nn.Embedding.from_pretrained(embeddings=text_embedding_vectors)
        # self.scalar = nn.Linear(in_features=self.input_dim, out_features=1, bias=False)

    def forward(self, x):


        # only the last layer parameters are updated, the others remain fixed.
        feature_extract = True

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--pretrained_state_dict_path', default='normal_transformer_finetuned.mdl', type=str)
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    print(args)

    # modelをロード
    model = torch.load('./raw_model/transformer.pth')
    # 学習済みのパラメータをロード
    pretrained_state_dict = torch.load(args.pretrained_state_dict_path, map_location=args.device)