# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/17 17:39
@Function:
"""
import gc
from sklearn import preprocessing

import pandas as pd

from sklearn.preprocessing import LabelEncoder


class Ad:

    def __init__(self, ad_data):
        self.ad_raw = ad_data.fillna('-1')
        self.ad_feature = pd.DataFrame()
        self.ad_feature['aid'] = self.ad_raw['aid']

        self.base_feature = ['advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType']

        for feat in self.base_feature:
            try:
                self.ad_feature[feat] = LabelEncoder().fit_transform(self.ad_raw[feat].apply(int))
            except:
                self.ad_feature[feat] = LabelEncoder().fit_transform(self.ad_raw[feat])

        print('------------ad base feature process over...')
        print()


if __name__ == '__main__':

    data = pd.read_csv('data/raw_data/adFeature.csv')
    ad = Ad(data)

    ad.ad_feature.to_csv('data/feature_data/clean_ad_feature.csv', index=False)
