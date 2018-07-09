# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/19 12:03
@Function:
"""
import collections

import gc
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd


class Feature2vec:

    def __init__(self, data, size=15):
        self.size = size
        data = data.fillna('-1')

        # self.feature = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
        self.feature = ['topic2']

        self.w2v_feature = pd.DataFrame()
        self.w2v_feature['uid'] = data['uid']
        for feat in self.feature:
            self.w2v_feature[feat] = data[feat]

    def get_feature(self):
        for feature in self.feature:
            print("this is feature:", feature)
            # self._select_topk(feature)
            self.w2v_feature[feature] = self.w2v_feature[feature].apply(lambda x: str(x).split(' '))
            model = Word2Vec(self.w2v_feature[feature], size=self.size, min_count=1, window=2, iter=5, workers=32)
            # model = Word2Vec.load('model/' + str(self.size) + feature +  '15_w2v.mod')
            model.save('data/w2v_model/w2v_' + str(self.size) + feature + '.mod')

            data_vec = []
            for row in self.w2v_feature[feature]:
                data_vec.append(self._base_word2vec(row, model))
            column_names = []
            for i in range(self.size):
                column_names.append(feature + str(i))

            data_vec = pd.DataFrame(data_vec, columns=column_names)
            data_vec['uid'] = self.w2v_feature['uid']
            data_vec = data_vec.round(4)
            data_vec.to_csv("data/w2v_feature/w2v_all_" + str(self.size) + feature + '.csv', index=False)
            gc.collect()


    def _base_word2vec(self, x, model):
        vec = np.zeros(self.size)
        # x = [word for word in x if model.__contains__(word)]
        for item in x:
            vec += model.wv[item]
        if len(x) == 0:
            return vec
        else:
            return vec / len(x)

    def _select_topk(self, feature):
        word_list = []
        for line in self.w2v_feature[feature]:
            words = str(line).split(' ')
            word_list += words
        result = collections.Counter(word_list)
        size = len(result)
        result = result.most_common(int(size))

        print(result[0], result[int(size * 0.01)], result[int(size * 0.05)], result[int(size * 0.1)],
              result[int(size * 0.2)], result[int(size * 0.3)], result[int(size * 0.4)], result[int(size * 0.5)])
        result = [res for res in result if 1000 < res[1] < 1000000]

        word_dict = {}
        for re in result:
            word_dict[re[0]] = 1

        self.w2v_feature[feature] = self.w2v_feature[feature].apply(
            lambda x: ' '.join([word for word in str(x).split(' ') if word_dict.__contains__(word)]))

    def get_topk(self):
        new_pd = pd.DataFrame()
        vector_feature = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
        # vector_feature = ['kw2']
        for feature in vector_feature:
            print("this is feature:", feature)

            self.data[feature] = self.data[feature].apply(lambda x: str(x).split(' '))
            word_dict = self.select_topk(self.data[feature])
            new_pd[feature] = self.data[feature].apply(
                lambda x: ' '.join([word for word in x if word_dict.__contains__(word)]))
            temp = new_pd[feature].apply(lambda x: len(str(x).split(' ')))
            print(temp.describe())

        new_pd.to_csv('./data/data_top50%feature.csv', index=False)


if __name__ == '__main__':
    data = pd.read_csv('data/raw_data/userFeature.csv')
    # data = pd.read_csv('data/data_5%.csv')
    word_vec = Feature2vec(data, size=20)
    word_vec.get_feature()
    # word_vec.w2v_feature.to_csv('data/raw_data/user_w2vFeature.csv', index=False)
    data = data.fillna('-1')

    # features = ['uid', 'interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
    # data = data[features]
    # print(data.shape)
    # all_ = data.shape[0]
    # for feat in features:
    #     count = data[data[feat] == '-1'].shape[0]
    #     print(feat, count / all_)
    # data.to_csv('data/feature_data/user_w2vFeature_all.csv', index=False)
