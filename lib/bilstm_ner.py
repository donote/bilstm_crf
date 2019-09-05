# encoding=utf-8

import os
import json
import torch
import torch.nn as nn
from torchcrf import CRF
from functools import partial

torch.manual_seed(12)


class BiLSTM_CRF_NER(nn.Module):
    def __init__(self, hyper):
        super(BiLSTM_CRF_NER, self).__init__()
        self.emb_size = hyper['emb_size']
        self.hidden_size = hyper['hidden_size']

        self.data_root = hyper['data_root']
        self.word2id = json.load(open(os.path.join(self.data_root, 'word_vocab.json')))
        self.vocab_size = len(self.word2id)
        self.bio2id = json.load(open(os.path.join(self.data_root, 'bio_vocab.json')))
        self.id2bio = [e[0] for e in sorted(self.bio2id.items(), key=lambda x: x[1])]
        self.bio_size = len(self.bio2id)
        self.gpu = hyper['gpu']

        self.word_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_size)
        self.lstm = nn.LSTM(self.emb_size, self.hidden_size // 2,
                            num_layers=1, bidirectional=True)

        self.encoder = nn.LSTM(input_size=self.emb_size,
                               hidden_size=self.hidden_size,
                               num_layers=2,
                               bidirectional=True,
                               batch_first=True)

        self.emission = nn.Linear(self.hidden_size, self.bio_size-1)  # last is <pad>
        self.crf = CRF(self.bio_size-1, batch_first=True)

    def forward(self, sample, is_train: bool):
        if self.gpu == -1:
            tokens = sample.tokens_id
            bio_gold = sample.bio_id
        else:
            tokens = sample.tokens_id.cuda(self.gpu)
            bio_gold = sample.bio_id.cuda(self.gpu)

        text_list = sample.text
        bio_tag = sample.bio

        mask = tokens != 0  #word_vocab['<pad>']  # batch x seq
        bio_mask = mask

        embedded = self.word_embedding(tokens)
        # pad没有mask掉 !
        o, h = self.encoder(embedded)
        o = (lambda a: sum(a) / 2)(torch.split(o, self.hidden_size, dim=2))

        emi = self.emission(o)

        output = {}
        crf_loss = 0
        if is_train:
            crf_loss = -self.crf(emi, bio_gold, mask=bio_mask, reduction='mean')
        else:
            decoded_tag = self.crf.decode(emissions=emi, mask=bio_mask)
            output['decoded_tag'] = [list(map(lambda x: self.id2bio[x], tags)) for tags in decoded_tag]
            output['gold_tags'] = bio_tag

        output['loss'] = crf_loss
        output['description'] = partial(self.description, output=output)
        return output

    @staticmethod
    def description(epoch, epoch_num, output):
        return "train loss: {:.2f}, epoch: {}/{}:".format(output['loss'].item(), epoch+1, epoch_num)

