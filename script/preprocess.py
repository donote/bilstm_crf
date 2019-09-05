# /usr/bin/env python
# encoding=utf8
# 对LAS标注的大段文本进行预处理，转为句子粒度
# 并划分train:dev:test == 8:1:1


import sys
import os
import random
import json
import codecs
import click
import re
from tqdm import tqdm


class Preprocessor(object):
    """
    输入：单行 string
    输出：预处理过后的 string
    """
    def __init__(self, remove_space=True, q2b=True, remove_empty_str=True):
        self.remove_space = remove_space
        self.q2b = q2b
        self.remove_empty_str = remove_empty_str

    @staticmethod
    def _segment(sentence):
        """分句"""
        pattern = r',|，|。|\?|\!|！|？'
        sents = re.split(pattern, sentence)
        return sents

    @staticmethod
    def _remove_space(sentence):
        """去空格"""
        return re.sub(r'\s', '', sentence)

    @staticmethod
    def _q2b(sentence):
        """全角转半角"""
        ret_sent = ''
        for uchar in sentence:
            inside_code = ord(uchar)
            # 全角空格直接转换
            if inside_code == 12288:
                inside_code = 32
            # 全角字符（除空格）根据关系转化
            elif inside_code >= 65281 and inside_code <= 65374:
                inside_code -= 65248
            ret_sent += chr(inside_code)
        return ret_sent

    def processing(self, ann_meta_data):
        """
        :param ann_meta_data:
        :return: 拆分为句子粒度，同时重置标注start/end下标
        """
        anns = ann_meta_data['annotation']
        text = ann_meta_data['text']
        urid = ann_meta_data['urid']
        if self.q2b:
            text = self._q2b(text)
        if self.remove_space:
            text = self._remove_space(text)

        for ann in anns:
            start, end, value = ann['start'], ann['end'], ann['value']
            if text[start:end] != value:  # 如果存在空格被替换掉，则会被丢弃
                sys.stderr('Data Format Error: {}'.format(ann_meta_data))
                continue

        # TODO 在做 q2b 和 remove_space 的时候会导致 pos 偏移，现在只能处理数据已经做个处理后的 pos，需要后面实现
        text_list = self._segment(text)
        sid, si = 0, 0
        for sent in text_list:
            sid += 1
            ei = si + len(sent)
            elem = {'text': sent, 'urid': urid, 'sid': sid, 'index': [si, ei], 'anns': []}
            for ann in anns:
                start, end, value = ann['start'], ann['end'], ann['value']
                if start >= si and end <= ei:
                    sub_ann = {'start': start-si, 'end': end-si, 'value': value}
                    elem['anns'].append(sub_ann)
            if len(sent.strip()) == 0:
                continue
            si = ei + 1
            yield elem
            #print(json.dumps(elem, ensure_ascii=False))


@click.command()
@click.option('-i', 'inputfile', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-o', 'outputfile', type=click.Path())
def text2sentence_anns(inputfile, outputfile):
    process = Preprocessor()
    out_fd = codecs.open(outputfile, mode='w', encoding='utf8')
    with codecs.open(inputfile, encoding='utf8') as fd:
        jdata = json.load(fd)
        for elem in tqdm(jdata, desc='processing'):
            for sub_elem in process.processing(elem):
                out_fd.write(json.dumps(sub_elem, ensure_ascii=False))
                out_fd.write('\n')

    # split samples to train/dev/test within 8:1:1
    result = gen_samples(outputfile)
    print('count of :\t train:{}\tdev:{}\ttest:{}'.format(result[0], result[1], result[2]))


def gen_samples(outputfile):
    # train:dev:test = 8:1:1
    output = os.path.dirname(outputfile)
    train_fd = codecs.open(os.path.join(output, 'train.txt'), mode='w', encoding='utf8')
    dev_fd = codecs.open(os.path.join(output, 'dev.txt'), mode='w', encoding='utf8')
    test_fd = codecs.open(os.path.join(output, 'test.txt'), mode='w', encoding='utf8')

    train_cnt, dev_cnt, test_cnt = 0, 0, 0
    random.seed(7)
    with codecs.open(outputfile, encoding='utf8') as fd:
        for line in fd:
            val = random.randint(1, 10)
            if val == 1:  # test
                test_fd.write(line.strip())
                test_fd.write('\n')
                test_cnt += 1
            elif val == 2:  # dev
                dev_fd.write(line.strip())
                dev_fd.write('\n')
                dev_cnt += 1
            else:   # train
                train_fd.write(line.strip())
                train_fd.write('\n')
                train_cnt += 1
    train_fd.close()
    dev_fd.close()
    test_fd.close()
    return train_cnt, dev_cnt, test_cnt


def unit_test():
    meta = {'annotation': [{'end': 51, 'start': 46, 'value': '阿帕替尼片'}],
            'text': u'入院完善检查,复查CT示病情进展,  于10.27转入我科,病理回示:(左肺)鳞癌。给予患者阿帕替尼片靶向治疗,配合消癌平及康莱特中药抗肿瘤治疗配合中药调理治疗,过程顺利。',
            'urid': '00023902fbb56e33a2601f4824f6dda7__index__20171023',
            }
    p = Preprocessor([], [], [])
    for e in p.processing(meta):
        print(e)


if __name__ == '__main__':
    #unit_test()
    text2sentence_anns()
