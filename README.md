# tencent_ad

### 腾讯社交广告算法大赛 Baseline

- baseline :
    - baseline_topk: 选择在interest kw topic等特征中出现频率topk的值，删除剩余的低频值

- 由于kw topic等id类特征繁多，并且发现很多kw或者topic成对出现：
    - 因此首先使用word2vec进行词向量构造，
    - 然后使用k-mean对词向量进行降维，对相似度极高的kw或者topic进行合并处理；再使用

- 缺陷：
    - 由于机器原因 词向量的维度太小，而id类特征太多，导致词向量无法对id进行很好的区分；
    - 数据量较大，kmean需要极大的内存开销，并且聚类时间较长；





