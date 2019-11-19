import numpy as np
from keras.layers import *
from keras.optimizers import Adam
from tqdm import tqdm
from keras_bert import Tokenizer, load_trained_model_from_checkpoint
import keras.backend as K
from keras.models import Model
from keras.callbacks import Callback
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score


class Lookahead(object):
    """Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).
    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model):
        """Inject the Lookahead algorithm for the given model.
        The following code is modified from keras's _make_train_function method.
        See: https://github.com/keras-team/keras/blob/master/keras/engine/training.py#L497
        """
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')

        model._check_trainable_weights_consistency()

        if model.train_function is None:
            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if model._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]
            fast_params = model._collected_trainable_weights

            with K.name_scope('training'):
                with K.name_scope(model.optimizer.__class__.__name__):
                    training_updates = model.optimizer.get_updates(
                        params=fast_params,
                        loss=model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]
                fast_updates = (model.updates +
                                training_updates +
                                model.metrics_updates)

                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))

                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(
                    inputs,
                    [model.total_loss] + model.metrics_tensors,
                    updates=fast_updates,
                    name='fast_train_function',
                    **model._function_kwargs)

                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R

                model.train_function = F


class Util(object):
    """
    工具器
    """

    def __init__(self):
        pass

    @staticmethod
    def _get_tokenizer(path):
        """
        获取tokenizer
        :param path:
        :return:
        """

        token_dict = {}
        with open(path) as reader:
            for line in reader:
                line = line.strip()
                token_dict[line] = len(token_dict)

        return Tokenizer(token_dict)

    @staticmethod
    def apply_multiple(input_, layers):
        """
        申请多个输入
        :param input_:
        :param layers:
        :return:
        """

        if not len(layers) > 1:
            raise ValueError('Layers list should contain more than 1 layer')
        else:
            agg_ = []
            for layer in layers:
                agg_.append(layer(input_))
            out_ = Concatenate()(agg_)

        return out_

    @staticmethod
    def get_model(bert_config_path, bert_checkpoint_path):
        """
        加载模型
        :param bert_config_path:
        :param bert_checkpoint_path:
        :return:
        """

        bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path)

        for l in bert_model.layers:
            l.trainable = True

        T1 = Input(shape=(None,))
        T2 = Input(shape=(None,))

        T = bert_model([T1, T2])

        T = Lambda(lambda x: x[:, 0])(T)
        T = Dense(units=64, activation='selu')(T)
        T = BatchNormalization()(T)
        output = Dense(2, activation='softmax')(T)
        model = Model([T1, T2], output)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])
        lookahead = Lookahead(k=5, alpha=0.5)  # 初始化Lookahead
        lookahead.inject(model)  # 插入到模型中
        model.summary()

        return model

    @staticmethod
    def predict(data, tokenizer, model):
        """
        模型预测
        :param data:
        :param tokenizer:
        :param model:
        :return:
        """
        prob = []
        val_x1, val_x2 = data
        for i in tqdm(range(len(val_x1))):
            achievements = val_x1[i]
            requirements = val_x2[i]

            t1, t1_ = tokenizer.encode(first=achievements, second=requirements)
            T1, T1_ = np.array([t1]), np.array([t1_])
            _prob = model.predict([T1, T1_])
            prob.append(_prob[0])

        return prob


class DataGenerator(object):
    """
    生成器
    """

    def __init__(self, tokenizer, data, batch_size=16, MAX_LEN=64):
        self.tokenizer = tokenizer
        self.data = data
        self.batch_size = batch_size
        self.MAX_LEN = MAX_LEN

        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            X1, X2, y = self.data
            idxs = list(range(len(self.data[0])))
            np.random.shuffle(idxs)
            T, T_, Y = [], [], []
            for c, i in enumerate(idxs):
                achievements = X1[i]
                requirements = X2[i]
                t, t_ = self.tokenizer.encode(first=achievements, second=requirements, max_len=self.MAX_LEN)
                T.append(t)
                T_.append(t_)
                Y.append(y[i])
                if len(T) == self.batch_size or i == idxs[-1]:
                    T = np.array(T)
                    T_ = np.array(T_)
                    Y = np.array(Y)
                    yield [T, T_], Y
                    T, T_, Y = [], [], []


class Evaluate(Callback):
    """
    评估器
    """

    def __init__(self, model, val_data, val_index, learning_rate, min_learning_rate, tokenizer, oof_train, fold=None):
        self.model = model
        self.score = []
        self.best = 0.
        self.early_stopping = 0
        self.val_data = val_data
        self.val_index = val_index
        self.predict = []
        self.lr = 0
        self.passed = 0
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.tokenizer = tokenizer
        self.oof_train = oof_train
        self.fold = fold

    def on_batch_begin(self, batch, logs=None):
        """
            第一个epoch用来warmup，第二个epoch把学习率降到最低
        :param batch:
        :param logs:
        :return:
        """
        if self.passed < self.params['steps']:
            self.lr = (self.passed + 1.) / self.params['steps'] * self.learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            self.lr = (2 - (self.passed + 1.) / self.params['steps']) * (self.learning_rate - self.min_learning_rate)
            self.lr += self.min_learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        """
        训练一轮结束
        :param epoch:
        :param logs:
        :return:
        """
        score, acc, f1 = self.evaluate()
        if score > self.best:
            self.best = score
            self.early_stopping = 0
            if self.fold is None:
                self.model.save_weights('bert.w')
            else:
                self.model.save_weights('bert{}.w'.format(self.fold))
        else:
            self.early_stopping += 1

    def evaluate(self):
        """
        评估
        :return:
        """
        self.predict = []
        prob = []
        val_x1, val_x2, val_y, val_cat = self.val_data
        for i in tqdm(range(len(val_x1))):
            achievements = val_x1[i]
            requirements = val_x2[i]

            t1, t1_ = self.tokenizer.encode(first=achievements, second=requirements)
            T1, T1_ = np.array([t1]), np.array([t1_])
            _prob = self.model.predict([T1, T1_])
            self.oof_train[self.val_index[i]] = _prob[0]
            self.predict.append(np.argmax(_prob, axis=1)[0] + 1)
            prob.append(_prob[0])

        score = 1.0 / (1 + mean_absolute_error(val_y + 1, self.predict))
        acc = accuracy_score(val_y + 1, self.predict)
        f1 = f1_score(val_y + 1, self.predict, average='macro')

        return score, acc, f1

    def get_oof_train(self):
        """
        获取 oof_train
        :return:
        """

        return self.oof_train
