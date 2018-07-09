# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/5/15 15:06
@Function:
"""
import random

import lightgbm as lgb
import time
from sklearn.metrics import auc, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import numpy as np


class Feature_selection:
    def __init__(self, data, base_feature, test_feature):

        self.data = data
        self.test_features = test_feature
        self.base_feature = base_feature
        self.best_features = []
        print('len base_feature: ', len(self.base_feature), '  len test feature: ', len(self.test_features))

    def _base_classifier(self):
        clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                 max_depth=-1, n_estimators=100, objective='binary',
                                 subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                 learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=20)
        return clf

    def _get_features_score(self, features):
        # 计算三次结果的最好得分作为最终的结果，尽量消除由随机带来的误差
        best_scores = []
        model = self._base_classifier()
        X_train, X_test, y_train, y_test = train_test_split(self.data[features].values, self.data['label'].values,
                                                            test_size=0.5)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1024)
        for i, (train_i, test_i) in enumerate(skf.split(X_train, y_train)):
            model.fit(X_train[train_i], y_train[train_i])
            y_pred = model.predict_proba(X_train[test_i])[:, 1]
            score = roc_auc_score(y_train[test_i], y_pred)
            best_scores.append(score)
        return np.mean(best_scores)

    def find_best_features(self):
        feat_num = len(self.test_features)

        try:
            for num_round in range(5):
                test_features = self.test_features[:]
                now_feature = self.base_feature[:]
                now_scores = self._get_features_score(self.base_feature)
                print('-------------now best features len: ', len(self.best_features), 'best scores: ', now_scores,
                      '-------------this is round 5 \ :', num_round)
                for i in range(feat_num):
                    start = time.time()
                    feat = test_features[random.randint(0, len(test_features) - 1)]
                    print('    --- this is feature:', feat, '  ---this is: ', i, ' / ', len(self.test_features))
                    now_feature.append(feat)
                    new_scores = self._get_features_score(now_feature)
                    # print('        --- now_scores:', now_scores, '----(now - best) score : ', new_scores - now_scores)
                    if (new_scores - now_scores) > 0.0001:
                        now_scores = new_scores
                        if feat not in self.best_features:
                            self.best_features.append(feat)
                            print('        ---now best features len: ', len(self.best_features), '   ---best scores: ',
                                  now_scores)
                    else:
                        now_feature.remove(feat)
                    test_features.remove(feat)
                    print('    ----time min:', (time.time() - start) / 60)
        except Exception as e:
            print('************************** Error ************************************')
            print(e)
            print('*********************************************************************')
        finally:
            print('       ---best features len: ', len(self.best_features))
            print(self.best_features)


def get_data():
    print('------------------------read data :')
    df_train = pd.read_csv('data/raw_data/train.csv')
    df_test = pd.read_csv('data/raw_data/test1.csv')
    df_train['label'] = df_train['label'].apply(lambda x: 0 if x == -1 else x)
    user_feature = pd.read_csv('data/feature_data/clean_user_feature.csv')
    ad_feature = pd.read_csv('data/feature_data/clean_ad_feature.csv')
    data = pd.concat([df_train, df_test])
    data = pd.merge(data, user_feature, on=['uid'], how='left')
    data = pd.merge(data, ad_feature, on=['aid'], how='left')
    print('user_feature.shape:', user_feature.shape)
    print('ad_feature.shape:', ad_feature.shape)

    # cross_feature = pd.read_csv('data/feature_data/cross_feature_probe.csv')
    # data = pd.merge(data, cross_feature, on=['aid', 'uid'], how='left')
    # print('cross_feature.shape:', cross_feature.shape)

    # cross_feature3 = pd.read_csv('data/feature_data/cross_feature3_probe.csv')
    # data = pd.merge(data, cross_feature3, on=['aid', 'uid'], how='left')
    # print('cross_feature3.shape:', cross_feature3.shape)

    # nlp_feature = pd.read_csv('data/feature_data/nlp_feature.csv')
    # data = pd.merge(data, nlp_feature, on=['uid'], how='left')
    # print('nlp_feature.shape:', nlp_feature.shape)

    kmeans_feature = pd.read_csv('data/feature_data/kmeans_feature.csv')
    data = pd.merge(data, kmeans_feature, on=['uid'], how='left')
    print('kmeans_feature.shape:', kmeans_feature.shape)

    # features = ['kw1', 'kw2', 'topic1', 'topic2']
    # for feat in features:
    #     kmeans_feature = pd.read_csv('data/w2v_feature/w2v_15' + feat + '.csv')
    #     data = pd.merge(data, kmeans_feature, on=['uid'], how='left')
    #     print('kmeans_feature.shape:', kmeans_feature.shape)

    train = data[data.label.notnull()]
    return train


if __name__ == '__main__':
    data = get_data()

    base_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                    'marriageStatus', 'advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId',
                    'productType', 'len_interest1', 'len_interest2', 'len_interest5', 'len_kw3', 'aid']

    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    remove_feature = ['uid', 'label']
    out_feature = base_feature + vector_feature + remove_feature

    test_features = [feat for feat in data.columns.tolist() if feat not in out_feature]
    print(test_features)
    feature_select = Feature_selection(data, base_feature, test_features)
    feature_select.find_best_features()
    # print(feature_select.best_features)