"""
For Las ann samples
"""

import os
import json
import codecs

from tqdm import tqdm
from collections import Counter


class BIO_Preprocessing(object):
    def __init__(self, hyper):
        self.hyper = hyper
        self.raw_data_root = hyper['raw_data_root']
        self.data_root = hyper['data_root']
        self.word_vocab = os.path.join(hyper['data_root'],
                                       hyper.get('word_vocab', 'word_vocab.json'))
        self.bio_vocab = os.path.join(hyper['data_root'],
                                      hyper.get('bio_vocab', 'bio_vocab.json'))
        self.pretrain_emb = os.path.join(hyper['data_root'],
                                         hyper.get('pretrain_emb', 'pretrain_emb.json'))
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

    def gen_bio_vocab(self):
        result = {'B': 0, 'I': 1, 'O': 2, '<pad>': 3}
        fd = codecs.open(self.bio_vocab, 'w', encoding='utf8')
        json.dump(result, fd, ensure_ascii=False, indent=4)
        print('Bio Vocab Set: {}'.format(len(result)))
        return result

    def gen_vocab(self, min_freq: int):
        source = os.path.join(self.raw_data_root, self.hyper['train'])
        target = self.word_vocab

        cnt = Counter()
        with codecs.open(source, 'r', encoding='utf8') as s:
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
        json.dump(result, codecs.open(target, 'w', encoding='utf8'), ensure_ascii=False, indent=4)
        print('Token Vocab Set: {}'.format(len(result)))
        return result

    def gen_pretrain_emb(self, vocab):
        """
        :param vocab: dict(word: idx)
        :function: 如果使用预训练emb，根据vocab先生成实际所需embedding
        """
        pretrain_path = self.hyper.get('pretrain', '')
        if not os.path.exists(pretrain_path):
            print('pretrain {} is not exists'.format(pretrain_path))
            return None

        import numpy as np
        w2v_model, w2v_dim = self.load_embed_use_gensim(pretrain_path)
        assert w2v_dim == self.hyper['emb_size']
        vocab_size = len(vocab)
        np.random.seed(137)
        # 预训练embedding中存在word则使用预训练数据，否则使用随机正态分布
        w2v_emb = np.random.normal(size=(vocab_size, w2v_dim)).astype('float32')
        unk_cnt = 0
        for k, v in vocab.items():
            if k in w2v_model.vocab:
                w2v_emb[v] = w2v_model[k]
            else:
                unk_cnt += 1

        # 写入文件
        import pickle
        pickle.dump(w2v_emb, codecs.open(self.pretrain_emb, 'wb'), protocol=2)
        print('\tOOV in Pretrain: {0}/{1}'.format(unk_cnt, vocab_size))
        return w2v_emb, unk_cnt

    def load_embed_use_gensim(self, path_embed):
        """
        读取预训练的embedding
        """
        from gensim.models.keyedvectors import KeyedVectors
        assert path_embed.endswith('bin') or path_embed.endswith('txt')
        binary = True if path_embed.endswith('bin') else False
        word_vectors = KeyedVectors.load_word2vec_format(path_embed, binary=binary)
        return word_vectors, word_vectors.vector_size

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
        with codecs.open(source, 'r', encoding='utf8') as s, codecs.open(target, 'w', encoding='utf8') as t:
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

