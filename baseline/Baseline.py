# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/18 17:49
@Function:
"""

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd
import lightgbm as lgb
from gensim.models.word2vec import Word2Vec


def base_word2vec(x, model, size):
    vec = np.zeros(size)
    x = [item for item in x if model.wv.__contains__(item)]

    for item in x:
        vec += model.wv[item]
    if len(x) == 0:
        return vec
    else:
        return vec / len(x)


def base_process(data):
    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']

    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    lbc = LabelEncoder()
    for feature in one_hot_feature:
        print("this is feature:", feature)
        try:
            data[feature] = lbc.fit_transform(data[feature].apply(int))
        except:
            data[feature] = lbc.fit_transform(data[feature])

    for feature in vector_feature:
        print("this is feature:", feature)
        data[feature] = data[feature].apply(lambda x: str(x).split(' '))
        model = Word2Vec(data[feature], size=10, min_count=1, iter=5, window=2)
        data_vec = []
        for row in data[feature]:
            data_vec.append(base_word2vec(row, model, size=10))
        column_names = []
        for i in range(10):
            column_names.append(feature + str(i))
        data_vec = pd.DataFrame(data_vec, columns=column_names)
        data = pd.concat([data, data_vec], axis=1)
        del data[feature]
    return data


def base_model(train, test, best_iter=100):
    col = [c for c in train if c not in ['uid', 'label']]
    X = train[col]
    y = train['label'].values
    print('------------------Training LGBM model--------------------------')
    lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
        num_leaves=40,
        max_depth=6,
        learning_rate=0.1,
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


def do_exp():
    print('-------------------- read data  -------------------------------------')
    # merge之后的data
    data = pd.read_csv('input/testData.csv')
    data = base_process(data)
    data.to_csv('input/data_0420.csv', index=False)
    print('-------------------- train and test data ----------------------------')
    train = data[data.label.notnull()]
    test = data[data.label.isnull()]
    base_model(train, test, best_iter=1000)


if __name__ == '__main__':
    do_exp()
    print('end...')