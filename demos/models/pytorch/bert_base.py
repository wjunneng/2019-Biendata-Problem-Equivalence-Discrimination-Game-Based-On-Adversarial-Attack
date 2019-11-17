# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from configurations.constant import Constant

TYPE = 'bert'


class BertBase(object):
    def __init__(self):
        self.configuration = Constant(type=TYPE).get_configuration()
        self.project_path = Constant(type=TYPE).get_project_path()

        self.train_set_path = os.path.join(self.project_path, self.configuration.train_data_csv_cache_path)
        self.dev_set_path = os.path.join(self.project_path, self.configuration.dev_set_path)

        self.bert_config_path = os.path.join(self.project_path, self.configuration.bert_config_path)
        self.bert_checkpoint_path = os.path.join(self.project_path, self.configuration.bert_checkpoint_path)
        self.bert_vocab_path = os.path.join(self.project_path, self.configuration.bert_vocab_path)

        self.MAX_LEN = self.configuration.MAX_LEN

        self.learning_rate = self.configuration.learning_rate
        self.min_learning_rate = self.configuration.learning_rate

    @staticmethod
    def label_process(x):
        if x == 0:
            return [1, 0]
        else:
            return [0, 1]

    def main(self):
        train = pd.read_csv(self.train_set_path)
        test = pd.read_csv(self.dev_set_path, sep='\t')

        train_achievements = train['question1'].values
        train_requirements = train['question2'].values
        labels = train['label'].values

        train['label'] = train['label'].apply(BertBase.label_process)
        label_cat = train['label'].values
        print('end!')

if __name__ == '__main__':
    BertBase().main()