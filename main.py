# /usr/bin/env python
# encoding=utf8

import os
import re
import json
from datetime import datetime
import argparse
import codecs
import torch
from tqdm import tqdm
from torch.optim import Adam, SGD

from lib.dataset import NERDataset, NERDataLoader
from lib.bilstm_ner import BiLSTM_CRF_NER
from lib.bio_preprocess import BIO_Preprocessing
from lib.F1_score import F1_ner_bio

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',
                    '-e',
                    type=str,
                    default='medical',
                    help='medical')
parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='prep',
                    help='prep|train|eval|predict')
args = parser.parse_args()


class Runner(object):
    def __init__(self, exp_name: str):
        self.exp_name = exp_name

        self.hyper = json.load(open(os.path.join('config/', self.exp_name+'.json'), 'r'))
        self.model_dir = self.hyper['saved_models']

        self.gpu = self.hyper['gpu']
        self.preprocessor = None
        self.ner_metrics = F1_ner_bio()
        self.optimizer = None
        self.model = None
        self.opt_lr = self.hyper.get('lr', 0.01)
        self.patient = self.hyper.get('patient', 100)
        self.re_parser = re.compile(r'BI*')

        self.eval_result_file = '{}.{}'.format(self.hyper.get('eval_result_file', '/tmp/bio_eval'),
                                               datetime.strftime(datetime.now(), '%Y-%m-%d'))
        self.predict_result_file = '{}.{}'.format(self.hyper.get('predict_result_file', '/tmp/bio_predict'),
                                               datetime.strftime(datetime.now(), '%Y-%m-%d'))

    def _optimizer(self, name, model):
        m = {'adam': Adam(model.parameters(), lr=self.opt_lr),
             'sgd': SGD(model.parameters(), lr=self.opt_lr)}
        return m[name]

    def _init_model(self):
        if self.gpu == -1:  # no gpu
            self.model = BiLSTM_CRF_NER(self.hyper)
        else:
            self.model = BiLSTM_CRF_NER(self.hyper).cuda(self.gpu)

    def preprocessing(self):
        self.preprocessor = BIO_Preprocessing(self.hyper)
        self.preprocessor.gen_all_dataset()
        self.preprocessor.gen_bio_vocab()
        self.preprocessor.gen_vocab(min_freq=1)

    def run(self, mode: str):
        # The Main Entrance Function
        if mode == 'prep':
            self.preprocessing()
        elif mode == 'train':
            self._init_model()
            self.optimizer = self._optimizer(self.hyper['optimizer'], self.model)
            self.train()
        elif mode == 'eval':
            self._init_model()
            self.load_model(epoch=self.hyper['evaluation_epoch'])
            self.evaluation()
        elif mode == 'predict':
            self._init_model()
            self.load_model(epoch=self.hyper['evaluation_epoch'])
            self.predict()
        else:
            raise ValueError('invalid mode')

    def load_model(self, epoch: int):
        model_filepath = os.path.join(self.model_dir, self.exp_name + '_' + str(epoch))
        model_data = torch.load(model_filepath)
        self.model.load_state_dict(model_data)

    def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        model_filepath = os.path.join(self.model_dir, self.exp_name + '_' + str(epoch))
        torch.save(self.model.state_dict(), model_filepath)

    def predict_result_save(self, fd, text_list, results, target=None, urid=None):
        """
        function: 将预测结果以spo格式写入文件
        text_list: tuple(str,)
        results: list(list(x, ), )
        """
        for i in range(len(text_list)):
            elem = {'text': text_list[i], 'predict': []}
            if isinstance(urid, tuple):
                elem['urid'] = urid[i]
            if isinstance(target, tuple):
                elem['target'] = target[i]
            bio = ''.join(results[i])
            for s in self.re_parser.finditer(bio):
                si, ei = s.start(), s.end()
                value = text_list[i][si:ei]
                e = {'start': si, 'end': ei, 'value': value}
                elem['predict'].append(e)
            fd.write(json.dumps(elem, ensure_ascii=False, indent=4))
            fd.write('\n====\n')

    def evaluation(self, dataloader=None, fd_eval=None):
        if not dataloader:
            dev_set = NERDataset(self.hyper, self.hyper['dev'])
            dataloader = NERDataLoader(dev_set, batch_size=self.hyper['eval_batch'], pin_memory=True, shuffle=False)

        self.model.eval()
        fd = codecs.open(self.eval_result_file, mode='w', encoding='utf8')

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)
                self.ner_metrics(output['gold_tags'], output['decoded_tag'])
                self.predict_result_save(fd, sample.text, output['decoded_tag'], target=sample.value, urid=sample.urid)

            ner_result = self.ner_metrics.get_metric_ner()
            eval_info = 'NER->' + ', '.join(["%s: %.4f" % (name[0], value) \
                                       for name, value in ner_result.items() if not name.startswith("_")])
            print(eval_info)
            if fd_eval:
                fd_eval.write(eval_info)
                fd_eval.write('\n')
                fd_eval.flush()
        fd.close()
        return ner_result['fscore']

    def train(self):
        train_set = NERDataset(self.hyper, self.hyper['train'])
        train_loader = NERDataLoader(train_set, batch_size=self.hyper['train_batch'], pin_memory=True)

        dev_set = NERDataset(self.hyper, self.hyper['dev'])
        dev_loader = NERDataLoader(dev_set, batch_size=self.hyper['eval_batch'], pin_memory=True, shuffle=False)

        if 'resume_model' in self.hyper and self.hyper['resume_model'] != 0:
            self.load_model(self.hyper['resume_model'])

        result_file_eval_tmp = '/tmp/bio_eval.{}.txt'.format(datetime.strftime(datetime.now(), '%Y-%m-%d'))
        fd_eval = codecs.open(result_file_eval_tmp, mode='w', encoding='utf8')

        patient_cnt = 0
        epoch_best, bio_f1_best = 0, 0.
        for epoch in range(self.hyper['epoch_num']):
            self.model.train()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for batch_idx, sample in pbar:
                self.optimizer.zero_grad()
                output = self.model(sample, is_train=True)
                loss = output['loss']
                loss.backward()
                self.optimizer.step()

                pbar.set_description(output['description'](
                    epoch, self.hyper['epoch_num']))

            # 注意这里不是连续eval，而是在print_epoch间隔时做的eval
            if (epoch + 1) % self.hyper['print_epoch'] == 0:
                bio_f1 = self.evaluation(dev_loader, fd_eval)
                if bio_f1 > bio_f1_best:
                    bio_f1_best = bio_f1
                    epoch_best = epoch + 1
                    print('====Best BIO F1 in Epoch {}===='.format(epoch_best))
                    self.save_model(epoch_best)
                    patient_cnt = 0
                else:
                    patient_cnt += 1
            if patient_cnt >= self.patient:
                break
        fd_eval.close()

    def predict(self):
        """
        只对text文本进行预测，无标注数据
        """
        test_set = NERDataset(self.hyper, self.hyper['test'], type='predict')
        loader = NERDataLoader(test_set, batch_size=self.hyper['eval_batch'], pin_memory=True)
        self.model.eval()

        fd = codecs.open(self.predict_result_file, mode='w', encoding='utf8')
        pbar = tqdm(enumerate(loader), total=len(loader))
        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)
                self.predict_result_save(fd, sample.text, output['decoded_tag'], target=sample.value, urid=sample.urid)

        fd.close()


if __name__ == "__main__":
    runner = Runner(exp_name=args.exp_name)
    runner.run(mode=args.mode)

