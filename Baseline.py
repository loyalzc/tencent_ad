# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/17 17:49
@Function:
"""

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd
import lightgbm as lgb


def base_marriage(x):
    x = str(x)
    if x == '5 13 10':
        return 1
    elif x == '15 10' or x == '8' or x == '6 13 10' or x == '13 15 10' or x == '13 10 9' or x == '2 13 10':
        return 2
    elif x == '12 13 10' or x == '3':
        return 3
    else:
        return 0


def base_education(x):
    x = int(x)
    if x == 0 or x == 1 or x == 2 or x == 3:
        return 1
    else:
        return 0


def base_age(x):
    x = int(x)
    if x == 0 or x == 1 or x == 3:
        return 1
    elif x == 2:
        return 2
    else:
        return 0


def base_ct(x):
    x_list = x.split(' ')
    x_list.sort()
    return ''.join(x_list)


def base_os(x):
    x = str(x)

    if x == 0 or x == 1:
        return 1
    else:
        return 0


def base_process(data):
    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']

    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    data['age_0'] = data['age'].apply(base_age)
    data['education_0'] = data['education'].apply(base_education)
    data['os_0'] = data['os'].apply(base_os)
    data['marriageStatus_0'] = data['marriageStatus'].apply(base_marriage)

    lbc = LabelEncoder()
    for feature in one_hot_feature:
        print("this is feature:", feature)
        try:
            data[feature] = lbc.fit_transform(data[feature].apply(int))
        except:
            data[feature] = lbc.fit_transform(data[feature])

    for feature in vector_feature:
        print("this is feature:", feature)
        data['len_' + feature] = data[feature].map(lambda x: len(str(x).split(' ')))
        del data[feature]
    return data


def base_model(train, test, best_iter=300):
    col = [c for c in train if c not in ['uid', 'aid', 'label']]
    X = train[col]
    y = train['label'].values
    print('------------------Training LGBM model--------------------------')
    lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
        num_leaves=40,
        max_depth=6,
        learning_rate=0.01,
        seed=2018,
        colsample_bytree=0.8,
        # min_child_samples=8,
        subsample=0.9,
        n_estimators=best_iter)
    lgb_model = lgb0.fit(X, y)

    print('----------------------predict result --------------------------')
    pred = lgb_model.predict_proba(test[col])[:, 1]
    test['score'] = pred
    test['score'] = test['score'].apply(lambda x: round(x, 7))

    result = test[['aid', 'uid', 'score']]
    result.to_csv('submission.csv', index=False)


def do_exp2():
    print('-------------------- read data  -------------------------------------')
    #
    data = pd.read_csv('input/testData.csv')
    data.fillna('-1')

    data = base_process(data)

    data.to_csv('input/data_0420.csv', index=False)
    print('-------------------- train and test data ----------------------------')
    train = data[data.label != -1]
    test2 = data[data.label == -1]
    base_model(train, test2, best_iter=1000)


if __name__ == '__main__':
    # do_exp()
    do_exp2()

    print('end....')