# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/17 17:49
@Function:
"""
import time

import numpy
import random
import pandas as pd
import scipy.special as special


class Nlp_feature:

    def __init__(self, nlp_feature, base_data=None, user_data=None, feature=None):
        """

        :param nlp_feature: 处理后生成的模型 每个用户一个
        :param base_data:  用户和广告的merge数据， 用于计算item的转化率
        :param user_data: 用户数据 用户根据 feature 生成对应的nlp_feature
        :param feature: 需要处理的特征列表
        """
        self.nlp_feature = nlp_feature
        self.base_data = base_data
        self.user_data = user_data
        self.feature = feature

    class HyperParam(object):
        """贡献平滑参数"""

        def __init__(self, alpha, beta):
            self.alpha = alpha
            self.beta = beta

        def sample_from_beta(self, alpha, beta, num, imp_upperbound):
            # 产生样例数据
            sample = numpy.random.beta(alpha, beta, num)
            I = []
            C = []
            for click_ratio in sample:
                imp = random.random() * imp_upperbound
                # imp = imp_upperbound
                click = imp * click_ratio
                I.append(imp)
                C.append(click)
            return pd.Series(I), pd.Series(C)

        def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
            # 更新策略
            for i in range(iter_num):
                new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
                if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                    break
                self.alpha = new_alpha
                self.beta = new_beta

        def __fixed_point_iteration(self, tries, success, alpha, beta):
            # 迭代函数
            sumfenzialpha = 0.0
            sumfenzibeta = 0.0
            sumfenmu = 0.0
            sumfenzialpha = (special.digamma(success + alpha) - special.digamma(alpha)).sum()
            sumfenzibeta = (special.digamma(tries - success + beta) - special.digamma(beta)).sum()
            sumfenmu = (special.digamma(tries + alpha + beta) - special.digamma(alpha + beta)).sum()

            return alpha * (sumfenzialpha / sumfenmu), beta * (sumfenzibeta / sumfenmu)

    def _nlp_feature_score(self, feat):
        item_count = {}  # 记录 item 出现的次数
        item_true_count = {}  # 记录item的转化率信息

        data_list = self.base_data[feat].values.tolist()
        # label_list = self.base_data['label'].values.tolist()
        for i, list in enumerate(data_list):
            items = list.split(' ')
            for item in items:
                if item_count.__contains__(item):
                    item_count[item] += 1
                else:
                    item_count[item] = 1
                if not item_true_count.__contains__(item):
                    item_true_count[item] = 1
                elif item_true_count.__contains__(item):
                    item_true_count[item] += 1
        dianji = []
        zhuanhua = []
        for item in item_true_count.keys():
            dianji.append(item_count[item])
            zhuanhua.append(item_true_count[item])
        # 获取平滑参数
        info = pd.DataFrame({'dianji': dianji, 'zhuanhua': zhuanhua})
        hyper = self.HyperParam(1, 1)
        hyper.update_from_data_by_FPI(info['dianji'], info['zhuanhua'], 1000, 0.00000001)

        data_socre = []
        # 使用平滑参数 平滑数据 （类似拉普拉斯平滑）
        data_list = self.user_data[feat].values.tolist()
        for i, list in enumerate(data_list):
            score = 1
            for item in list.split(' '):
                if not item_true_count.__contains__(item):
                    score += 0
                else:
                    score = score * (1 - (item_true_count[item] + hyper.alpha) / (
                            item_count[item] + hyper.alpha + hyper.beta))
            data_socre.append(score)

        self.nlp_feature[feat + '_score'] = data_socre
        self.nlp_feature[feat + '_score'] = self.nlp_feature[feat + '_score'].round(7)
        # self.nlp_feature[feat + '_score'] = pd.cut(self.nlp_feature[feat + '_score'], 10, labels=range(10))

    def get_nlp_data(self):
        print('-----------------nlp feature:')
        for feat in self.feature:
            print('   ----this is feature: ', feat)
            self._nlp_feature_score(feat)

    def get_nlp_discr(self):
        print('-----------------nlp get_nlp_discr:')
        for feat in self.feature:
            print('   ----this is feature: ', feat)
            self.nlp_feature[feat + '_score'] = pd.cut(self.nlp_feature[feat + '_score'], 10, labels=range(10))
            # self.nlp_feature[feat + '_score'] = pd.qcut(self.nlp_feature[feat + '_score'], 10, labels=range(10))


if __name__ == '__main__':
    df_train = pd.read_csv('data/raw_data/train.csv')
    df_test = pd.read_csv('data/raw_data/test1.csv')
    df_train['label'] = df_train['label'].apply(lambda x: 0 if x == -1 else x)
    df_userFeature = pd.read_csv('data/raw_data/userFeature.csv')
    df_adFeature = pd.read_csv('data/raw_data/adFeature.csv')
    data = pd.concat([df_train, df_test])
    data = pd.merge(data, df_userFeature, on=['uid'], how='left')
    data = pd.merge(data, df_adFeature, on=['aid'], how='left')

    data = data.fillna('-1')
    df_userFeature = df_userFeature.fillna('-1')

    nlp_feature = pd.DataFrame()
    nlp_feature['uid'] = df_userFeature['uid']

    # nlp_feature = pd.read_csv('data/feature_data/nlp_feature.csv')

    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    # vector_feature = ['kw1', 'kw2', 'topic1', 'topic2']
    # vector_feature = ['kw2']
    start = time.time()
    # nlp = Nlp_feature(nlp_feature, feature=vector_feature)
    nlp = Nlp_feature(nlp_feature, data, df_userFeature, vector_feature)
    nlp.get_nlp_data()
    # nlp.get_nlp_discr()
    nlp.nlp_feature.to_csv('data/feature_data/nlp_feature.csv', index=False)

    # nlp.nlp_feature.to_csv('data/feature_data/nlp_feature_cut.csv', index=False)
    print('cost time :', (time.time() - start) / 60)

