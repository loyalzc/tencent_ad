# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/17 17:49
@Function:
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def get_samell_data():
    print('--------------------read data---------------------------------------')
    df_train = pd.read_csv('data/train_small_20%.csv')
    # df_train = pd.read_csv('data/raw_data/train.csv')
    # df_test = pd.read_csv('data/raw_data/test1.csv')
    df_userFeature = pd.read_csv('data/raw_data/userFeature.csv')
    df_adFeature = pd.read_csv('data/raw_data/adFeature.csv')
    df_train['label'] = df_train['label'].apply(lambda x: 0 if x == -1 else x)

    print('--------------------merge data-------------------------------------')
    # data = pd.concat([df_train, df_test])
    data = pd.merge(df_train, df_userFeature, on=['uid'], how='left')
    data = pd.merge(data, df_adFeature, on=['aid'], how='left')

    # print(df_train.label.count())
    # print(df_train[df_train.label == 1].count())
    # train_y = df_train.pop('label')
    # #
    # X_train, X_test, y_train, y_test = train_test_split(df_train.values, train_y.values, test_size=0.2)
    # #
    # data = pd.DataFrame(X_test, columns=['aid', 'uid'])
    # data['label'] = y_test
    # print(len(data.aid.unique()))
    data.to_csv('data/data_20%.csv', index=False)


if __name__ == '__main__':
    # data = pd.read_csv('data/raw_data/userFeature.csv')
    # data = pd.read_csv('data/raw_data/train.csv')
    # get_samell_data()
    data = pd.read_csv('data/data_5%_k_base.csv', encoding='utf-8')
    data = data.fillna('-1')
    colunms = data.columns.values
    row = data.shape[0]
    test_data = data.sample(int(row * 0.2))
    all_ = pd.concat([data, test_data])
    train_data = all_.drop_duplicates(keep=False)
    print(data.shape, test_data.shape, train_data.shape)
    # data.to_csv('data/data_5%_fillna.csv', index=False)
    # print(test_data[test_data.label==1].count())
    train_data.to_csv('train_5%_data.csv', index=False)
    test_data.to_csv('test_5%_data.csv', index=False)
