# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/17 17:39
@Function:
"""
import gc
from sklearn import preprocessing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


class Cross_feature:

    def __init__(self, base_data, cross_feature):
        self.base_data = base_data
        self.cross_feature = cross_feature

    def cross_2feature(self, feat1, feat2):
        item = self.base_data.groupby(feat1, as_index=False)['uid'].agg({feat1 + '_count': 'count'})
        self.base_data = pd.merge(self.base_data, item, on=[feat1], how='left')

        itemcnt = self.base_data.groupby([feat1, feat2], as_index=False)['uid'].agg({feat1 + feat2: 'count'})
        self.base_data = pd.merge(self.base_data, itemcnt, on=[feat1, feat2], how='left')
        self.cross_feature[feat1 + '_' + feat2 + '_prob'] = self.base_data[feat1 + feat2] / self.base_data[feat1 + '_count']
        self.cross_feature[feat1 + '_' + feat2 + '_prob'] = pd.qcut(self.cross_feature[feat1 + '_' + feat2 + '_prob'], 10, duplicates='drop')
        # self.cross_feature[feat1 + '_' + feat2 + '_prob'] = self.cross_feature[feat1 + '_' + feat2 + '_prob'].round(7)

        del self.base_data[feat1 + '_count']
        del self.base_data[feat1 + feat2]
        gc.collect()
        print('    ----cross feature: %s  and %s' % (feat1, feat2))


    def cross_3feature(self, feat1, feat2, feat3):
        item = self.base_data.groupby([feat1, feat2], as_index=False)['uid'].agg({feat1 + feat2 + '_count': 'count'})
        self.base_data = pd.merge(self.base_data, item, on=[feat1, feat2], how='left')

        itemcnt = self.base_data.groupby([feat1, feat2, feat3], as_index=False)['uid'].agg({feat1 + feat2 + feat3: 'count'})
        self.base_data = pd.merge(self.base_data, itemcnt, on=[feat1, feat2, feat3], how='left')
        self.cross_feature[feat1 + '_' + feat2 + '_' + feat3 + '_prob'] = \
            self.base_data[feat1 + feat2 + feat3] / self.base_data[feat1 + feat2 + '_count']
        # self.cross_feature[feat1 + '_' + feat2 + '_' + feat3 + '_prob'] = \
        # pd.cut(self.cross_feature[feat1 + '_' + feat2 + '_' + feat3 + '_prob'], 10, labels=range(10))

        self.cross_feature[feat1 + '_' + feat2 + '_' + feat3 + '_prob'] = \
            self.cross_feature[feat1 + '_' + feat2 + '_' + feat3 + '_prob'].round(7)

        del self.base_data[feat1 + feat2 + '_count']
        del self.base_data[feat1 + feat2 + feat3]
        gc.collect()
        print('    ----cross feature: %s  %s  and %s' % (feat1, feat2, feat3))


    def base_cross(self):
        print('-------------------------------cross features----------------------------------')

        feature1 = ['aid', 'advertiserId', 'adCategoryId',  'creativeId', 'productId']
        feature2 = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'ct', 'marriageStatus']

        for feat1 in feature1:
            for feat2 in feature2:
                self.cross_2feature(feat1, feat2)

    def base_cross3(self):
        print('-------------------------------cross features----------------------------------')

        feature1 = ['aid', 'advertiserId', 'adCategoryId']
        feature2 = ['LBS', 'age', 'carrier', 'consumptionAbility']
        feature3 = ['education', 'gender', 'ct', 'marriageStatus']

        for feat1 in feature1:
            for feat2 in feature2:
                for feat3 in feature3:
                    self.cross_3feature(feat1, feat2, feat3)

    def combine_feature(self):
        """用户直接拼接特征"""
        print('---------------combine_feature--------------------')
        features = ['aid', 'advertiserId', 'age', 'gender']
        for col in features:
            self.base_data[col] = self.base_data[col].astype(str)

        self._combine('aid', 'age')
        self._combine('advertiserId', 'age')
        self._combine('aid', 'gender')
        self._combine('advertiserId', 'gender')

    def _combine(self, feat1, feat2):
        print('    ----cross feature: %s and %s' % (feat1, feat2))
        self.base_data[feat1 + '_' + feat2] = self.base_data[feat1] + '_' + self.base_data[feat2]
        self.cross_feature[feat1 + '_' + feat2] = LabelEncoder().fit_transform(self.base_data[feat1 + '_' + feat2])
        self.cross_feature[feat1 + '_' + feat2] = self.cross_feature[feat1 + '_' + feat2].apply(int)


if __name__ == '__main__':
    print('--------------------cross feature----------------------')
    # train = pd.read_csv('data/raw_data/train.csv')
    # test = pd.read_csv('data/raw_data/test1.csv')
    # user_feature = pd.read_csv('data/feature_data/clean_user_feature.csv')
    # ad_feature = pd.read_csv('data/feature_data/clean_ad_feature.csv')
    #
    # base_data = pd.concat([train, test])
    # base_data = pd.merge(base_data, user_feature, on=['uid'], how='left')
    # base_data = pd.merge(base_data, ad_feature, on=['aid'], how='left')
    #
    # # base_data = pd.read_csv('data/data_5%_fillna.csv')
    #
    # cross_feature = pd.DataFrame()
    # cross_feature[['aid', 'uid']] = base_data[['aid', 'uid']]
    # cross = Cross_feature(base_data, cross_feature)
    # cross.base_cross()
    # cross.base_cross3()
    # cross.cross_feature.to_csv('data/feature_data/cross_feature_qcut.csv', index=False)



    user_feature = pd.read_csv('data/feature_data/cross_feature_qcut.csv')
    cols = user_feature.columns.values
    cols.remove('aid')
    cols.remove('uid')

    for feat in cols:
        user_feature[feat] = pd.qcut(user_feature[feat], 5, duplicates='drop')
        user_feature[feat] = LabelEncoder().fit_transform(user_feature[feat])

        user_feature.to_csv('data/feature_data/cross_feature_qcut.csv', index=False)
    print('end....')
