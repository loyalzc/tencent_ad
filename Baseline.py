# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/17 17:39
@Function:
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

from Tencent_AD2018 import feature_ad
from Tencent_AD2018 import util_models


def do_exp_with_w2vFeature():
    print('--------------------read data---------------------------------------')
    df_train = pd.read_csv('data/raw_data/train.csv')
    df_test = pd.read_csv('data/raw_data/test1.csv')
    df_userFeature = pd.read_csv('data/clean_userFeature.csv')
    df_adFeature = pd.read_csv('data/raw_data/adFeature.csv')
    df_train['label'] = df_train['label'].apply(lambda x: 0 if x == -1 else x)

    # print('--------------------concat w2v features ----------------------------')
    # w2v_featrure = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
    # # w2v_featrure = ['kw2']
    # # w2v_featrure = []
    # for feat in w2v_featrure:
    #     print("------this is feature: ", feat)
    #     df_w2vfeat = pd.read_csv('data/topk_w2v_feat/w2v_' + feat + '.csv')
    #     if df_w2vfeat.shape[0] == df_userFeature.shape[0]:
    #         df_w2vfeat = df_w2vfeat.round(2)
    #         df_userFeature = pd.concat([df_userFeature, df_w2vfeat], axis=1)
    #     else:
    #         print("************ " + feat + " Shape Error...  **********")

    # print('--------------------kmeans_feature    --------------------------------')
    # kmeans_feature = pd.read_csv('data/word2vec_feat/kmeans20.csv')
    # if kmeans_feature.shape[0] == df_userFeature.shape[0]:
    #     # kmeans_feature = kmeans_feature[['kw1', 'kw2', 'topic1', 'topic2']]
    #     df_userFeature = pd.concat([df_userFeature, kmeans_feature], axis=1)
    #
    # print('--------------------len_feature    --------------------------------')
    # len_feature = pd.read_csv('data/topk_w2v_feat/len_feature.csv')
    # if len_feature.shape[0] == df_userFeature.shape[0]:
    #     df_userFeature = pd.concat([df_userFeature, len_feature], axis=1)
    # else:
    #     print("************ len_feature Shape Error...  *************")

    print('--------------------merge data-------------------------------------')
    data = pd.concat([df_train, df_test])
    data = pd.merge(data, df_userFeature, on=['uid'], how='left')
    data = pd.merge(data, df_adFeature, on=['aid'], how='left')

    # features = ['aid', 'label', 'uid', 'age', 'gender', 'marriageStatus', 'education'
    #             , 'consumptionAbility', 'LBS', 'ct', 'os', 'carrier', 'house', 'kmeans_appIdAction', 'kmeans_appIdInstall', 'kmeans_interest1'
    #             , 'kmeans_interest2', 'kmeans_interest3', 'kmeans_interest4'
    #             , 'kmeans_interest5', 'kmeans_kw1', 'kmeans_kw2', 'kmeans_kw3', 'kmeans_topic1'
    #             , 'kmeans_topic2', 'kmeans_topic3', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize'
    #             , 'adCategoryId', 'productId', 'productType']
    # for col in features:
    #     data[col] = data[col].astype(str)

    print('--------------------cross feature----------------------------------')

    data = feature_ad.ad_base_process(data)
    # data = user.user_ad_feature(data)

    print('--------------------train model -----------------------------------')
    print(data.label.unique())
    print(data.columns.values)
    print(data.shape)
    train = data[data.label.notnull()]
    test = data[data.label.isnull()]

    # del train['uid']
    # train_y = train.pop('label')
    # train_x, test_x, traint_y, test_y = train_test_split(train, train_y, test_size=0.2, random_state=2018)
    # print('-------------------- train model ------------------------------------')
    # best_iter = models.lgbCV(train_x, test_x, traint_y, test_y)

    util_models.base_model(train, test, best_iter=1500)


def do_exp2_with_adFeature():
    print('-------------------- read data  -------------------------------------')
    data = pd.read_csv('data/data_user_aid.csv')

    print('-------------------- train and test data ----------------------------')
    print(data.label.unique())
    train = data[data.label.notnull()]
    del train['uid']
    train_y = train.pop('label')
    train_x, test_x, traint_y, test_y = train_test_split(train, train_y, test_size=0.2, random_state=2018)
    print('-------------------- train model ------------------------------------')
    best_iter = util_models.lgbCV(train_x, test_x, traint_y, test_y)
    # test2 = data[data.label.isnull()]
    # models.base_model(train, test2, best_iter=1000)


if __name__ == '__main__':
    # do_exp_with_w2vFeature()
    # do_exp2_with_adFeature()
    data = pd.read_csv('data/feature_data/clean_user_feature2.csv')
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    features = ['kmeans_20_' + feat for feat in vector_feature]
    cols = ['uid'] + features
    data = data[cols]
    data.to_csv('data/feature_data/kmeans_feature.csv', index=False)

    print('end....')