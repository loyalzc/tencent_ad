# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/19 12:03
@Function:
"""

import pickle
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans


def get_kmeans_feature(test, n_clusters=10):
    print('-------------------------get_kmeans_feature--------')
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    vector_feature = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2']

    data = test[vector_feature]
    data['uid'] = test['uid']
    for i, feature in enumerate(vector_feature):
        print("------this is feature: ", feature)
        model = Word2Vec.load('data/w2v_model/w2v_20' + feature + '.mod')
        word_vector = model.wv.syn0
        if word_vector.shape[0] < 1000:
            num_clusters = 25
        elif word_vector.shape[0] < 50000:
            num_clusters = 250
        else:
            num_clusters = 500

        kmeans_clustering = KMeans(n_clusters=num_clusters, n_jobs=-1)
        idx = kmeans_clustering.fit_predict(word_vector)
        word_centroid_map = dict(zip(model.wv.index2word, idx))

        filename = 'data/w2v_model/' + feature + 'word_to_idx_map.pickle'
        with open(filename, 'bw') as f:
            pickle.dump(word_centroid_map, f)

        def get_idx(x, word_centroid_map):
            id_set = set([str(word_centroid_map[word]) for word in x if word_centroid_map.__contains__(word)])
            return ' '.join(id_set)

        data[feature] = data[feature].apply(lambda x: str(x).split(' '))
        data[feature] = data[feature].apply(lambda x: get_idx(x, word_centroid_map))
    return data


if __name__ == '__main__':
    data = pd.read_csv('data/raw_data/userFeature.csv')
    # test = pd.read_csv('data/raw_data/userFeature_kmeans.csv')
    data = get_kmeans_feature(data)
    data.to_csv('data/raw_data/userFeature_kmeans.csv', index=False)
    # print(data.shape)
    # model = Word2Vec.load('data/w2v_model/interest115_w2v.mod')
    # print(model.most_similar('11'))
    print('end...')

