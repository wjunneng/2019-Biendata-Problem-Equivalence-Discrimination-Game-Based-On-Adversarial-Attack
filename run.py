import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from sklearn.model_selection import StratifiedKFold

"摘自https://zhuanlan.zhihu.com/p/82737301"
import pandas as pd
import numpy as np
import keras.backend as K
from configurations.constant import Constant

from demos.models.keras.util import DataGenerator, Evaluate
from demos.models.keras.util import Util

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
        tokenizer = Util._get_tokenizer(path=self.bert_vocab_path)

        train = pd.read_csv(self.train_set_path)
        test = pd.read_csv(self.dev_set_path, sep='\t')
        train_achievements = train['question1'].values
        train_requirements = train['question2'].values
        labels = train['label'].values

        train['label'] = train['label'].apply(BertBase.label_process)
        labels_cat = list(train['label'].values)
        labels_cat = np.array(labels_cat)
        test_achievements = test['question1'].values
        test_requirements = test['question2'].values
        print(train.shape, test.shape)

        oof_train = np.zeros((len(train), 2), dtype=np.float32)
        oof_test = np.zeros((len(test), 2), dtype=np.float32)

        ind = np.array(list(range(train.shape[0])))

        # 设置随机种子
        np.random.seed(42)
        np.random.shuffle(ind)

        # ################################### 多折
        nfolds = 5
        skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)

        evaluator = None
        oof_train = np.zeros((len(train), 2), dtype=np.float32)
        oof_test = np.zeros((len(test), 2), dtype=np.float32)
        for fold, (train_index, valid_index) in enumerate(skf.split(train_achievements, labels)):
            x1 = train_achievements[train_index]
            x2 = train_requirements[train_index]
            y = labels_cat[train_index]
            val_x1 = train_achievements[valid_index]
            val_x2 = train_requirements[valid_index]
            val_y = labels[valid_index]
            val_cat = labels_cat[valid_index]
            train_D = DataGenerator(tokenizer=tokenizer, data=[x1, x2, y], batch_size=128, MAX_LEN=self.MAX_LEN)
            model = Util.get_model(bert_config_path=self.bert_config_path,
                                   bert_checkpoint_path=self.bert_checkpoint_path)
            evaluator = Evaluate(model=model, val_data=[val_x1, val_x2, val_y, val_cat], val_index=valid_index,
                                 learning_rate=self.learning_rate, min_learning_rate=self.min_learning_rate,
                                 tokenizer=tokenizer, oof_train=oof_train, fold=fold)
            model.fit_generator(train_D.__iter__(),
                                steps_per_epoch=len(train_D),
                                epochs=3,
                                callbacks=[evaluator]
                                )
            model.load_weights('bert{}.w'.format(fold))
            oof_test += Util.predict(data=[test_achievements, test_requirements], tokenizer=tokenizer, model=model)
            K.clear_session()
        oof_test /= nfolds

        # ################################### 单折
        # train_index, valid_index = ind[:int(len(ind) * 0.8)], ind[int(len(ind) * 0.8):]
        # x1 = train_achievements[train_index]
        # x2 = train_requirements[train_index]
        # y = labels_cat[train_index]
        # val_x1 = train_achievements[valid_index]
        # val_x2 = train_requirements[valid_index]
        # val_y = labels[valid_index]
        # val_cat = labels_cat[valid_index]
        # train_D = DataGenerator(tokenizer=tokenizer, data=[x1, x2, y], batch_size=24, MAX_LEN=self.MAX_LEN)
        # model = Util.get_model(bert_config_path=self.bert_config_path, bert_checkpoint_path=self.bert_checkpoint_path)
        # evaluator = Evaluate(model=model, val_data=[val_x1, val_x2, val_y, val_cat], val_index=valid_index,
        #                      learning_rate=self.learning_rate, min_learning_rate=self.min_learning_rate,
        #                      tokenizer=tokenizer, oof_train=oof_train, fold=None)
        # model.fit_generator(train_D.__iter__(),
        #                     steps_per_epoch=len(train_D),
        #                     epochs=5,
        #                     callbacks=[evaluator]
        #                     )
        # model.load_weights('bert.w')
        # test = pd.DataFrame(Util.predict(data=[test_achievements, test_requirements], tokenizer=tokenizer, model=model))
        # K.clear_session()

        test.to_csv('test_pred.csv', index=False)
        test.head(), test.shape
        train = pd.DataFrame(evaluator.get_oof_train())
        train.to_csv('train_pred.csv', index=False)

        pred = pd.read_csv('test_pred.csv').values
        pred = pred.argmax(axis=1)
        sub = pd.DataFrame()
        sub['pred'] = list(pred)
        sub.to_csv('result.csv', sep='\t', header=None)


if __name__ == '__main__':
    BertBase().main()
