# -*- coding: utf-8 -*
import os
import random
import pandas as pd
from xml.dom.minidom import parse

from configurations.constant import Constant

TYPE = 'bert'


class Process(object):
    def __init__(self):
        self.project_path = Constant(type=TYPE).get_project_path()
        self.configuration = Constant(type=TYPE).get_configuration()

        self.dev_set_path = os.path.join(self.project_path, self.configuration.dev_set_path)
        self.train_set_xml_data = os.path.join(self.project_path, self.configuration.train_set_path)
        self.train_data_cache_path = os.path.join(self.project_path, self.configuration.train_data_cache_path)
        self.test_data_cache_path = os.path.join(self.project_path, self.configuration.test_data_cache_path)
        self.dev_data_cache_path = os.path.join(self.project_path, self.configuration.dev_data_cache_path)

    @staticmethod
    def generate_train_data_pair(equ_questions, not_equ_questions):
        a = [x + "\t" + y + "\t" + "0" for x in equ_questions for y in not_equ_questions]
        # 包含重复值,只是顺序不同 结果：100298条数据
        # b = [x + "\t" + y + "\t" + "1" for x in equ_questions for y in equ_questions if x != y]
        # 不包含重复值 结果：81080条数据
        b = []
        for s_index in range(len(equ_questions) - 1):
            for e_index in range(s_index + 1, len(equ_questions)):
                b.append(equ_questions[s_index] + "\t" + equ_questions[e_index] + "\t" + "1")

        return a + b

    @staticmethod
    def write_train_data(train_file, dev_file, pairs):
        random.shuffle(pairs)

        train_pairs = pairs[:int(len(pairs) * 0.8)]
        dev_pairs = pairs[int(len(pairs) * 0.8):]

        with open(train_file, mode="w") as file:
            for pair in train_pairs:
                file.write(pair + "\n")

        with open(dev_file, mode='w') as file:
            for pair in dev_pairs:
                file.write(pair + "\n")

    def parse_train_data(self):
        """
        构造训练集
        :return:
        """
        pair_list = []
        doc = parse(self.train_set_xml_data)
        collection = doc.documentElement
        for i in collection.getElementsByTagName("Questions"):
            # if i.hasAttribute("number"):
            #     print ("Questions number=", i.getAttribute("number"))
            EquivalenceQuestions = i.getElementsByTagName("EquivalenceQuestions")
            NotEquivalenceQuestions = i.getElementsByTagName("NotEquivalenceQuestions")
            equ_questions = EquivalenceQuestions[0].getElementsByTagName("question")
            not_equ_questions = NotEquivalenceQuestions[0].getElementsByTagName("question")
            equ_questions_list, not_equ_questions_list = [], []
            for q in equ_questions:
                try:
                    tmp = q.childNodes[0].data.strip().strip('？').strip('?').strip('。').strip('.').strip('!').strip('！')
                    equ_questions_list.append(tmp)
                except:
                    continue
            for q in not_equ_questions:
                try:
                    tmp = q.childNodes[0].data.strip().strip('？').strip('?').strip('。').strip('.').strip('!').strip('！')
                    not_equ_questions_list.append(tmp)
                except:
                    continue
            pair = Process.generate_train_data_pair(equ_questions_list, not_equ_questions_list)
            pair_list.extend(pair)
        print("All pair count=", len(pair_list))

        # 保存结果
        Process.write_train_data(self.train_data_cache_path, self.dev_data_cache_path, pair_list)

        return pair_list

    def generate_test_data(self):
        """
        生成测试集
        :return:
        """
        dev_csv = pd.read_csv(self.dev_set_path, encoding='utf_8_sig', sep='\t')
        question1 = dev_csv['question1']
        question2 = dev_csv['question2']

        with open(self.test_data_cache_path, 'w', encoding='utf_8_sig') as file:
            for index in range(dev_csv.shape[0]):
                file.write(question1[index] + '\t' + question2[index] + '\n')


if __name__ == "__main__":
    # 生成训练集和验证集
    configuration = Constant(type=TYPE).get_configuration()
    project_path = Constant(type=TYPE).get_project_path()
    train_data_csv_cache_path = os.path.join(project_path, configuration.train_data_csv_cache_path)
    pair_list = Process().parse_train_data()
    question1 = []
    question2 = []
    label = []
    for pair in pair_list:
        pair = pair.split('\t')
        question1.append(pair[0])
        question2.append(pair[1])
        label.append(pair[2])
    df = pd.DataFrame()
    df['question1'] = question1
    df['question2'] = question2
    df['label'] = label
    df.to_csv(train_data_csv_cache_path, index=False, encoding='utf-8')

    # 生成测试集
    Process().generate_test_data()
