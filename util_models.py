# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/17 18:07
@Function:
"""
import datetime

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
import json

class Model:

    def __init__(self, train_x, train_y, feaures_name, test_x=None, res=None):
        """
        :param train_x:
        :param train_y:
        :param test_x: 测试集特征
        :param res: 预测的信息
        :param feaures_name: 特征名称：用户输出重要程度
        """
        self.features = feaures_name
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.res = res

        self.base_model = self._base_model()

    def feature_impt_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.train_x, self.train_y, test_size=0.2)

        print('-------------------feature_impt_model:')
        self.base_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test,  y_test)],
                            eval_metric='auc', early_stopping_rounds=100)
        best_iter = self.base_model.best_iteration_
        print("   ---best_iter:", best_iter)

        predictors = [i for i in self.features]
        feat_imp = pd.Series(self.base_model.feature_importances_, predictors).sort_values(ascending=False)
        print(feat_imp)
        now = datetime.datetime.now()
        now = now.strftime('%m-%d-%H-%M')
        feat_imp.to_csv('data/feature_data/feature_impt' + str(now) + '.csv')

    def kfold_model(self, n_folds=3):
        print('------------------get kfold_model result --------------------------')
        self.train_x, X_test, self.train_y, y_test = train_test_split(self.train_x, self.train_y, test_size=0.002)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1024)
        for i, (train_i, test_i) in enumerate(skf.split(self.train_x, self.train_y)):
            print('---fold: ', i)
            self.base_model.fit(self.train_x[train_i], self.train_y[train_i], eval_metric='auc',
                                eval_set=[(self.train_x[train_i], self.train_y[train_i]),
                                          (self.train_x[test_i], self.train_y[test_i])],
                                early_stopping_rounds=100)

            pred = self.base_model.predict_proba(self.test_x, num_iteration=-1)[:, 1]
            self.res['probe_' + str(i)] = pred

        print('----------------------predict result --------------------------')
        self.res.to_csv('result_cv.csv', index=False)
        self.res['score'] = self.res['probe_0']
        for i in range(1, n_folds):
            self.res['score'] += self.res['probe_' + str(i)]
        self.res['score'] = self.res['score'] / n_folds
        self.res['score'] = self.res['score'].apply(lambda x: round(x, 7))

        result = self.res[['aid', 'uid', 'score']]
        now = datetime.datetime.now()
        now = now.strftime('%m-%d-%H-%M')
        result.to_csv('lgb_kfold_' + str(now) + '.csv', index=False)

    def _base_model(self):
        base_model = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=31, reg_alpha=0.01, reg_lambda=0.05,
            max_depth=6, n_estimators=1500, objective='binary',
            subsample=0.9, colsample_bytree=0.8, subsample_freq=1,
            learning_rate=0.1, min_child_weight=50, random_state=2018, n_jobs=-1
        )
        return base_model
