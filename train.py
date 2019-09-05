# /bin/usr/env python
# encoding=utf8

import sys
from model.bilstm_crf.bilstm_crf import BiLSTM_CRF
from model.bilstm_crf.bilstm_ner import BiLSTM_NER
from model.bilstm_crf.configs import EMBEDDING_DIM, HIDDEN_DIM, tag_to_ix
from model.bilstm_crf.dataset import NERDataset, NERDataLoader
from model.bilstm_crf.utils import read_json_file, get_ann, build_vocab, prepare_sequence
from model.bilstm_crf.preprocess import Preprocessor
from tqdm import tqdm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import random
import time
import pandas as pd

from model.bilstm_crf.configs import CONF
from model.bilstm_crf.configs import epoch_num

def preprocess(conf=CONF):
    pass


def main():
    data_path = './data/feiai_total.json'
    js = read_json_file(data_path)
    js = js[:10]
    word_to_ix, id2word = build_vocab(js)

    urids = [data['urid'] for data in js]
    texts = [data['text'] for data in js]
    anns = [get_ann(data['annotation']) for data in js]
    assert len(urids) == len(texts) == len(anns)

    preprocessor = Preprocessor(urids, texts, anns)
    # 划分句子粒度作为样本，需要对ner在句子中的位置进行转换
    # 这种预处理可作为独立的一步处理
    urids, texts, anns = preprocessor.run()
    dataset = NERDataset(urids, texts, anns, word_to_ix, tag_to_ix)
    loader = NERDataLoader(dataset)

    model = BiLSTM_NER(vocab_size=len(word_to_ix),
                       tag_to_ix=tag_to_ix,
                       embedding_dim=EMBEDDING_DIM,
                       hidden_dim=HIDDEN_DIM)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


    # # Check predictions before training
    # with torch.no_grad():
    #     print('prechecking START...')
    #     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    #     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    #     print('x', precheck_sent)
    #     print('y', precheck_tags)
    #     print('pred', model(precheck_sent))
    #     print('prechecking DONE...')

    # # Make sure prepare_sequence from earlier in the LSTM section is loaded
    # print('\ntraining START ...')
    for epoch in range(epoch_num):
        print('epoch {}: '.format(epoch), end='')
        model.train()
        start_time = time.time()
        running_loss = 0.0
        for idx_batch, batch in tqdm(enumerate(loader), desc='train at epoch {}'.format(epoch),
                                     total=len(loader), mininterval=1, leave=True, file=sys.stdout):
            optimizer.zero_grad()
            output = model(batch, is_train=True)
            loss = output['loss']
            running_loss += loss
            loss.backward()
            optimizer.step()
        print('{}'.format(output['description'](epoch, epoch_num)))

        end_time = time.time()
        eval_train = evaluate(train_data, word_to_ix)
        eval_test = evaluate(test_data, word_to_ix)
        print('epoch {} train-f1 {} dev-f1 {} loss {} time {:.1f}'
              .format(epoch, eval_train, eval_test, running_loss, end_time - start_time))


if __name__ == '__main__':
    main()