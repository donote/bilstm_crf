# coding: utf-8
import os
import json
from _functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence


class NERDataset(Dataset):

    def __init__(self, hyper, dataset, type='train|eval'):
        self.data_root = hyper['data_root']
        self.word2id = json.load(open(os.path.join(self.data_root, 'word_vocab.json')))
        self.vocab_size = len(self.word2id)
        self.bio2id = json.load(open(os.path.join(self.data_root, 'bio_vocab.json')))
        self.id2bio = [e[0] for e in sorted(self.bio2id.items(), key=lambda x: x[1])]
        self.bio_size = len(self.bio2id)
        self.gpu = hyper['gpu']
        self.type = type

        self.unk = self.word2id['<oov>']
        self.pad = self.word2id['<pad>']

        self.urid_list = []
        self.text_list = []
        self.bio_list = []
        self.value_list = []
        self.word_id_list = []
        self.bio_id_list = []

        self.data_buffer(filepath=os.path.join(self.data_root, dataset))

    def data_buffer(self, filepath):
        with open(filepath, 'r') as fd:
            for line in fd:
                line = line.strip("\n")
                instance = json.loads(line)
                self.urid_list.append(instance.get('urid', 'none'))
                self.text_list.append(instance['text'])
                self.word_id_list.append(torch.tensor([self.word2id.get(c, self.unk) for c in instance['text']]))
                if self.type == 'predict':
                    self.bio_list.append([])
                    self.bio_id_list.append([])
                    self.value_list.append([])
                else:
                    self.bio_list.append(instance['bio'])
                    self.bio_id_list.append(torch.tensor([self.bio2id[c] for c in instance['bio']]))
                    self.value_list.append(instance['anns'])

    def __getitem__(self, idx):
        sample_ret = (self.urid_list[idx], len(self.text_list[idx]), self.text_list[idx], \
                      self.word_id_list[idx], self.bio_list[idx], self.bio_id_list[idx], self.value_list[idx])
        return sample_ret

    def __len__(self):
        return len(self.text_list)


class Batch_reader(object):
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.tokens_id, _ = self._pad_sequence(transposed_data[3])
        # for predict mode ...
        sum_len_bio = sum([len(sz) for sz in transposed_data[5]])
        if sum_len_bio:
            self.bio_id, _ = self._pad_sequence(transposed_data[5], pad_value=3)
        else:  # predict mode ...
            self.bio_id = torch.tensor(0)

        self.length = transposed_data[1]
        self.text = transposed_data[2]
        self.bio = transposed_data[4]
        self.urid = transposed_data[0]
        self.value = transposed_data[6]

    @staticmethod
    def _pad_sequence(tokensid, pad_value=0):
        packed_sequence = pack_sequence(tokensid, enforce_sorted=False)
        padded_packed_sequence, lengths = pad_packed_sequence(packed_sequence, batch_first=True, padding_value=0)
        return padded_packed_sequence, lengths


def collate_handle(batch):
    return Batch_reader(batch)


NERDataLoader = partial(DataLoader, collate_fn=collate_handle, pin_memory=True, shuffle=False)


if __name__ == '__main__':
    data = [torch.tensor([3, 4, 5, 6, 9]),
            torch.tensor([3, 4, 5]),
            torch.tensor([3, 4])]

    pad_packed, length = Batch_reader._pad_sequence(data)
    print(pad_packed)
    print(length)
