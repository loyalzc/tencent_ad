# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/17 17:39
@Function:
"""

import pandas as pd
import numpy as np
from Tencent_AD2018.util_models import Model

select_cross2_feature = ['aid', 'uid', 'adCategoryId_LBS_prob', 'aid_age_prob', 'advertiserId_ct_prob', 'adCategoryId_marriageStatus_prob',
                  'aid_marriageStatus_prob', 'productId_LBS_prob',
 'aid_gender_prob', 'aid_ct_prob', 'productId_age_prob', 'advertiserId_gender_prob', 'creativeId_consumptionAbility_prob', 'creativeId_gender_prob',
  'aid_carrier_prob', 'productId_ct_prob', 'adCategoryId_consumptionAbility_prob', 'aid_education_prob', 'adCategoryId_age_prob',
  'adCategoryId_ct_prob', 'productId_carrier_prob', 'adCategoryId_education_prob', 'adCategoryId_gender_prob',
  'creativeId_marriageStatus_prob', 'advertiserId_age_prob', 'creativeId_age_prob', 'advertiserId_carrier_prob',
  'advertiserId_consumptionAbility_prob', 'creativeId_carrier_prob']


def do_exp():
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

    cross_feature = pd.read_csv('data/feature_data/cross_feature_probe.csv')
    cross_feature = cross_feature[select_cross2_feature]
    data = pd.merge(data, cross_feature, on=['aid', 'uid'], how='left')
    print('cross_feature.shape:', cross_feature.shape)

    # cross_feature3 = pd.read_csv('data/feature_data/cross_feature3_probe.csv')
    # data = pd.merge(data, cross_feature3, on=['aid', 'uid'], how='left')
    # print('cross_feature3.shape:', cross_feature3.shape
    # nlp_feature = pd.read_csv('data/feature_data/nlp_feature.csv')
    # data = pd.merge(data, nlp_feature, on=['uid'], how='left')
    # print('nlp_feature.shape:', nlp_feature.shape)

    # kmeans_feature = pd.read_csv('data/feature_data/kmeans_feature.csv')
    # data = pd.merge(data, kmeans_feature, on=['uid'], how='left')
    # print('kmeans_feature.shape:', kmeans_feature.shape)

    features = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']
    for feat in features:
        kmeans_feature = pd.read_csv('data/w2v_feature/w2v_all_20' + feat + '.csv')
        data = pd.merge(data, kmeans_feature, on=['uid'], how='left')
        print('kmeans_feature.shape:', kmeans_feature.shape)

    train = data[data.label.notnull()]
    test = data[data.label.isnull()]
    res = test[['aid', 'uid']]
    print('train.shape:', train.shape)
    print('test.shape:', test.shape)
    remove_cols = ['uid', 'label']
    cols = [col for col in train if col not in remove_cols]
    train_y = train['label'].values
    train_x = train[cols].values
    test_x = test[cols].values

    print('train.shape:', np.shape(train_x))
    print('test.shape:', np.shape(test_x))
    print('------------------------train model:')
    model = Model(train_x, train_y, cols, test_x, res)
    model.kfold_model()
    print('-----------end...')


if __name__ == '__main__':
    do_exp()
    # data = pd.read_csv('result_cv0512.csv')
