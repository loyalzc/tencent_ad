# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/17 17:49
@Function:
"""

from MLFeatureSelection import FeatureSelection as FS
from sklearn.metrics import log_loss
import lightgbm as lgbm
import pandas as pd
import numpy as np

def select_user_feature():
    data = pd.read_csv('clean_user_feature.csv')

def select_cross_feature():
    data = pd.read_csv('data/feature_data/cross_feature_probe.csv')
    cols = ['aid', 'uid', 'aid_LBS_prob', 'aid_age_prob',
            'adCategoryId_education_prob', 'creativeId_consumptionAbility_prob']
    data = data[cols]

    data.to_csv('data/feature_data/selected_cross2_feat.csv', index=False)


def select_cross3_feature():
    data = pd.read_csv('data/feature_data/cross_feature3_probe.csv')
    cols = ['aid', 'uid', 'aid_LBS_education_prob', 'aid_age_gender_prob',
            'aid_age_ct_prob', 'aid_consumptionAbility_gender_prob', 'advertiserId_LBS_gender_prob']

    data = data[cols]

    data.to_csv('data/feature_data/selected_cross3_feat.csv', index=False)


if __name__ == '__main__':
    select_cross_feature()
    select_cross3_feature()

