# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm

from configurations.constant import Constant

TYPE = 'bert'


class BertBase(object):
    def __init__(self):
        self.configuration = Constant(type=TYPE).get_configuration()
        self.project_path = Constant(type=TYPE).get_project_path()

        self.bert_config_path = self.configuration.bert_config_path
        self.bert_checkpoint_path = self.bert_checkpoint_path
        self.bert_vocab_path = self.bert_vocab_path