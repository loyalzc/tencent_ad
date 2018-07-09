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
        # self.vector_feature = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
        self.vector_feature = ['interest1']

    def get_vector_kmeans_feature(self, n_clusters=2, file_path='data/w2v_feature/w2v_all_20'):
        """使用词向量对用户进行聚类操作，作为用户特征"""
        print('------------get_vector_kmeans_feature start...')
        for i, feat in enumerate(self.vector_feature):
            print("    ---get_kmeans_feature: ", feat)
            df_w2vfeat = pd.read_csv(file_path + feat + '.csv')
            temp = df_w2vfeat['uid']
            del df_w2vfeat['uid']
            df_w2vfeat = df_w2vfeat.values
            k_means = KMeans(n_clusters=n_clusters, n_jobs=-1).fit(df_w2vfeat)
            label = k_means.labels_
            # print(np.shape(label))
            if len(label) == self.user_feature.shape[0]:
                temp['kmeans_' + str(n_clusters) + "_" + feat] = label
                self.user_feature = pd.merge(self.user_feature, temp, on=['uid'], how='left')
            else:
                print('*****************kmeans shape error!!!*****************')

        print('------------get_vector_kmeans_feature end...')
        print()



if __name__ == '__main__':
    user_feature = pd.read_csv('data/feature_data/clean_user_feature.csv')
    # vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4',
    #                   'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    user_feature = user_feature['uid']
    user = User(user_feature)
    user.get_vector_kmeans_feature()
    user.user_feature.to_csv('data/feature_data/w2v_kmeans_feature.csv', index=False)
