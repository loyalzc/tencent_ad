# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/5/7 20:09
@Function: transform data to ffm format
"""

import hashlib


def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16) % (nr_bins - 1) + 1


def gen_hashed_fm_feats(feats, nr_bins=int(1e+6)):
    feats = ['{0}:{1}:1'.format(field - 1, hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats


def get_data():
    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'aid', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4',
                      'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    drop_feature = ['uid', 'label']
    print("reading data")
    f = open('testData.csv.csv', 'r')
    line = f.readline().strip()
    features = line.split(',')

    print(features)
    dict = {}
    num = 0
    for line in f:
        datas = line.strip().split(',')
        for i, d in enumerate(datas):
            if not dict.__contains__(features[i]):
                dict[features[i]] = []
            dict[features[i]].append(d)
        num += 1

    f.close()

    print("transforming data")
    ftrain = open('data/testtest.ffm', 'w')

    for i in range(num):
        feats = []
        for j, f in enumerate(one_hot_feature, 1):
            field = j
            print('-----------dict[f][i]:', field, dict[f][i])
            feats.append((field, f + '_' + dict[f][i]))

        for j, f in enumerate(vector_feature, 1):
            field = j + len(one_hot_feature)
            xs = dict[f][i].split(' ')
            print('-----------xs:', xs)
            for x in xs:
                feats.append((field, f + '_' + x))

        feats = gen_hashed_fm_feats(feats)
        ftrain.write(dict['label'][i] + ' ' + ' '.join(feats) + '\n')
        # print(dict['label'][i] + ' ' + ' '.join(feats) + '\n')

    ftrain.close()


if __name__ == '__main__':
    get_data()