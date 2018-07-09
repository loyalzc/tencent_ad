# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/17 17:39
@Function:
"""

from scipy import sparse

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import lightgbm as lgb

import numpy as np

import warnings

from Tencent_AD2018.tencent.models import base_model, lgbCV

warnings.filterwarnings("ignore")


def LGB_predict(train_x, train_y, test_x, res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1500, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc', early_stopping_rounds=100)
    res['score'] = clf.predict_proba(test_x)[:, 1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('data/submit_ffm_05_10.csv', index=False)

    return clf


def get_one_hot_feature(data):

    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']
    vector_feature = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    print(data.label.unique())
    train = data[data.label.notnull()]
    test = data[data.label.isnull()]

    print(train.label.unique())
    print(test.label.unique())
    print(data.shape, train.shape, test.shape)

    train_y = train.pop('label').values

    del train['uid']
    res = test[['aid', 'uid']]
    test = test.drop('label', axis=1)
    clean_feature = [feat for feat in train if feat not in vector_feature]
    vector_feature = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
    print(clean_feature)
    train_x = train[['creativeSize']].values
    test_x = test[['creativeSize']].values

    enc = OneHotEncoder()
    print('-----one-hot prepared:')
    for feature in clean_feature:
        print('     ---feature: ', feature)
        enc.fit(data[feature].values.reshape(-1, 1))
        train_a = enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('-----cv prepared:')
    cv = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    cv = CountVectorizer()
    for feature in vector_feature:
        print('     ---feature: ', feature)
        # data[feature] =data[feature].apply(str)
        cv.fit(data[feature])
        train_a = cv.transform(train[feature])
        test_a = cv.transform(test[feature])
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))

    # sparse.save_npz("train.npz", train)
    # sparse.save_npz("train_y.npz", train_y)
    # sparse.save_npz("test_x.npz", train_y)

    # print(np.shape(train_x), np.shape(train_y), np.shape(test_x))
    return train_x, train_y, test_x, res


if __name__ == '__main__':
    print('------------------------read data :')
    df_train = pd.read_csv('data/raw_data/train.csv')
    df_test = pd.read_csv('data/raw_data/test2.csv')
    df_train['label'] = df_train['label'].apply(lambda x: 0 if x == -1 else x)
    user_feature = pd.read_csv('data/feature_data/clean_user_feature.csv')
    ad_feature = pd.read_csv('data/feature_data/clean_ad_feature.csv')
    data = pd.concat([df_train, df_test])
    data = pd.merge(data, user_feature, on=['uid'], how='left')
    data = pd.merge(data, ad_feature, on=['aid'], how='left')
    print('user_feature.shape:', user_feature.shape)
    print('ad_feature.shape:', ad_feature.shape)

    # cross_feature = pd.read_csv('data/feature_data/cross_feature.csv')
    # data = pd.merge(data, cross_feature, on=['aid', 'uid'], how='left')
    # print('cross_feature.shape:', cross_feature.shape)

    # cross_feature3 = pd.read_csv('data/feature_data/cross_feature3_probe.csv')
    # data = pd.merge(data, cross_feature3, on=['aid', 'uid'], how='left')
    # print('cross_feature3.shape:', cross_feature3.shape)

    # nlp_feature = pd.read_csv('data/feature_data/nlp_feature.csv')
    # data = pd.merge(data, nlp_feature, on=['uid'], how='left')
    # print('nlp_feature.shape:', nlp_feature.shape)

    # one_hot_feature = pd.read_csv('data/raw_data/userFeature.csv')
    # vector_feature = ['uid', 'interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
    #
    # one_hot_feature = one_hot_feature[vector_feature]
    # one_hot_feature = one_hot_feature.fillna('-1')
    # data = pd.merge(data, one_hot_feature, on=['uid'], how='left')
    # print('one_hot_feature.shape:', one_hot_feature.shape)

    one_hot_feature = pd.read_csv('data/raw_data/userFeature_kmeans.csv')
    one_hot_feature = one_hot_feature.fillna('-1')
    data = pd.merge(data, one_hot_feature, on=['uid'], how='left')
    print('one_hot_feature.shape:', one_hot_feature.shape)

    # features = ['kw1', 'kw2', 'topic1', 'topic2']
    # for feat in features:
    #     kmeans_feature = pd.read_csv('data/w2v_feature/w2v_15' + feat + '.csv')
    #     data = pd.merge(data, kmeans_feature, on=['uid'], how='left')
    #     print('kmeans_feature.shape:', kmeans_feature.shape)

    # data = pd.read_csv('data/raw_data/data_5%.csv')
    print('--------------------one_hot data-----------------------------------')
    train_X, train_y, test_X, res = get_one_hot_feature(data)
    print('--------------------train lgb model -------------------------------')
    # LGB_predict(train_x, train_y, test_x, res)
    train_x, test_x, train_y, test_y = train_test_split(train_X, train_y, test_size=0.002, random_state=2018)

    # lgbCV(train_x, test_x, train_y, test_y, 'null')

    base_model(train_x, train_y, test_X, res)

    # FM = pylibfm.FM(num_factors=500, num_iter=10, verbose=True, task="classification",
    #                 initial_learning_rate=0.01, learning_rate_schedule="optimal")
    # FM.fit(train_x, train_y)

    print('end....')