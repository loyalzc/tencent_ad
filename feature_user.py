# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/17 17:39
@Function:
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class User:
    """用户的基本特征"""
    def __init__(self, user_feature, user_data=None):

        self.user_feature = user_feature
        self.base_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender',
                             'house', 'os', 'ct', 'marriageStatus']

        self.vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4',
                               'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
        if user_data != None:
            print('   -----do base user feature')
            for feat in self.base_feature:
                try:
                    self.user_feature[feat] = LabelEncoder().fit_transform(user_data[feat].apply(int))
                except:
                    self.user_feature[feat] = LabelEncoder().fit_transform(user_data[feat])

            for feat in self.vector_feature:
                self.user_feature['len_' + feat] = user_data[feat].apply(lambda x: len(str(x).split(' ')))
                self.user_feature['len_' + feat] = pd.cut(self.user_feature['len_' + feat], 5, labels=range(5))
        print('------------user base feature process over...')
        print('user feature shape: ', user_feature.shape)

    def get_vector_kmeans_feature(self, n_clusters=20, file_path='data/w2v_feature/w2v_all_15'):
        """使用词向量对用户进行聚类操作，作为用户特征"""
        print('------------get_vector_kmeans_feature start...')
        for i, feat in enumerate(self.vector_feature):
            print("    ---get_kmeans_feature: ", feat)
            df_w2vfeat = pd.read_csv(file_path + feat + '.csv')
            k_means = KMeans(n_clusters=n_clusters, n_jobs=-1).fit(df_w2vfeat)
            label = k_means.labels_
            # print(np.shape(label))
            if len(label) == self.user_feature.shape[0]:
                self.user_feature['kmeans_' + str(n_clusters) + "_" + feat] = label
            else:
                print('*****************kmeans shape error!!!*****************')
        print('------------get_vector_kmeans_feature end...')
        print()

    def get_base_kmeans_feature(self, n_clusters=30):
        """对base的特征进行one-hot之后再做用户聚类，找到聚类特征"""

        print('------------get_base_kmeans_feature start...')
        one_hot_feature = pd.DataFrame()
        for feat in self.base_feature:
            one_hot_feature[feat] = OneHotEncoder().fit_transform(self.user_feature[feat])
        k_means = KMeans(n_clusters=n_clusters, n_jobs=10).fit(one_hot_feature)
        label = k_means.labels_

        self.user_feature['kmeans_base'] = label

        print('------------get_base_kmeans_feature end...')
        print()


if __name__ == '__main__':
    # userFeature = pd.read_csv('data/raw_data/userFeature.csv')
    # userFeature = userFeature.fillna('-1')
    # user_feature = pd.DataFrame()
    # user_feature['uid'] = userFeature['uid']

    user_feature = pd.read_csv('data/feature_data/clean_user_feature.csv')
    # vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4',
    #                   'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    # for feat in vector_feature:
    #
    #     user_feature['len_' + feat] = pd.cut(user_feature['len_' + feat], 5, labels=range(5))
    user_feature = user_feature['aid']
    user = User(user_feature)
    user.get_vector_kmeans_feature()
    # user.get_base_kmeans_feature()
    # print(user_feature.shape)
    # len_feature = ['len_interest1', 'len_interest2', 'len_interest5', 'len_kw3']
    # for feat in len_feature:
    #     print('this is ', feat)
    #     user_feature[feat] = pd.cut(user_feature[feat], 5, labels=range(5))
    #     # user_feature[feat] = LabelEncoder().fit_transform(user_feature[feat])
    # print('write..')
    # user_feature.to_csv('data/feature_data/clean_user_feature2.csv', index=False)
    # print('end')



