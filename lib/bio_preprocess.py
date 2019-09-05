"""
For Las ann samples
"""

import os
import json

from tqdm import tqdm
from collections import Counter


class BIO_Preprocessing(object):
    def __init__(self, hyper):
        self.hyper = hyper
        self.raw_data_root = hyper['raw_data_root']
        self.data_root = hyper['data_root']
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

    def gen_bio_vocab(self):
        result = {'B': 0, 'I': 1, 'O': 2, '<pad>': 3}
        fd = open(os.path.join(self.data_root, 'bio_vocab.json'), 'w')
        json.dump(result, fd, ensure_ascii=False, indent=4)
        print('Bio Vocab Set: {}'.format(len(result)))

    def gen_vocab(self, min_freq: int):
        source = os.path.join(self.raw_data_root, self.hyper['train'])
        target = os.path.join(self.data_root, 'word_vocab.json')

        cnt = Counter()
        with open(source, 'r') as s:
            for line in tqdm(s, desc='Generate Token Vocab'):
                line = line.strip("\n")
                if not line:
                    return None
                instance = json.loads(line)
                text = list(instance['text'])
                cnt.update(text)

        result = {'<pad>': 0, '<oov>': 1}
        i = len(result)
        for k, v in cnt.items():
            if v > min_freq:
                result[k] = i
                i += 1
        json.dump(result, open(target, 'w'), ensure_ascii=False, indent=4)
        print('Token Vocab Set: {}'.format(len(result)))

    def _read_line(self, line: str):
        line = line.strip("\n")
        if not line:
            return None

        instance = json.loads(line)
        text = instance['text']
        anns = instance['anns']

        bio = ['O'] * len(text)
        for e in anns:
            start, end = e['start'], e['end']
            assert end <= len(text)
            bio[start] = 'B'
            for i in range(start + 1, end):
                bio[i] = 'I'

        instance['bio'] = bio
        return json.dumps(instance, ensure_ascii=False)

    def _gen_one_dataset(self, dataset):
        source = os.path.join(self.raw_data_root, dataset)
        target = os.path.join(self.data_root, dataset)
        cnt = 0
        with open(source, 'r') as s, open(target, 'w') as t:
            for line in s:
                newline = self._read_line(line)
                if newline is not None:
                    cnt += 1
                    t.write(newline)
                    t.write('\n')
        return cnt

    def gen_all_dataset(self):
        print('Processing Train...')
        cnt_train = self._gen_one_dataset(self.hyper['train'])
        print('Processing Dev...')
        cnt_dev = self._gen_one_dataset(self.hyper['dev'])
        print('Processing Test...')
        cnt_test = self._gen_one_dataset(self.hyper['test'])
        print('train_set: {}, dev_set: {}, test_set: {}'.format(cnt_train, cnt_dev, cnt_test))

