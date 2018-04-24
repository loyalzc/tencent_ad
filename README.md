# tencent_ad

### 腾讯社交广告算法大赛 Baseline

大佬已经做出来了高Baseline的代码：https://github.com/YouChouNoBB/2018-tencent-ad-competition-baseline

无奈渣渣电脑基本跑不出来大神的结果，所以只好默默尝试别的出路

考虑interest kw topic此类特征太多，one-hot直接维数爆掉，所以采用了word2vec方法降维；没有调参的情况下基本可以达到one-hot的得分

- baseline :
- baseline_topk: 选择在interest kw topic等特征中出现频率topk的值，删除剩余的低频值

#### 另附interest kw topic各个维度的长度，以便选择word2vec的size

#### 用户特征


                    len_appIdAction len_appIdInstall len_interest1  len_interest2
            count     1.106480e+07      1.106480e+07   1.106480e+07   1.106480e+07   
            mean      1.137803e+00      3.306016e+00   1.294338e+01   4.164523e+00   
            std       1.732410e+00      2.864749e+01   8.972224e+00   4.244111e+00   
            min       1.000000e+00      1.000000e+00   1.000000e+00   1.000000e+00   
            25%       1.000000e+00      1.000000e+00   6.000000e+00   1.000000e+00   
            50%       1.000000e+00      1.000000e+00   1.200000e+01   2.000000e+00   
            75%       1.000000e+00      1.000000e+00   1.900000e+01   6.000000e+00   
            max       5.370000e+02      9.200000e+02   3.800000e+01   3.200000e+01   

 
                    len_interest3  len_interest4  len_interest5       len_kw1
            count   1.106480e+07   1.106480e+07   1.106480e+07  1.106480e+07   
            mean    1.168589e+00   1.050987e+00   1.515969e+01  4.392344e+00   
            std     1.136084e+00   4.851396e-01   1.185373e+01  1.350022e+00   
            min     1.000000e+00   1.000000e+00   1.000000e+00  1.000000e+00   
            25%     1.000000e+00   1.000000e+00   1.000000e+00  5.000000e+00   
            50%     1.000000e+00   1.000000e+00   1.500000e+01  5.000000e+00   
            75%     1.000000e+00   1.000000e+00   2.300000e+01  5.000000e+00   
            max     1.000000e+01   1.000000e+01   8.600000e+01  5.000000e+00   

                       len_kw2       len_kw3    len_topic1    len_topic2    len_topic3  
            count   1.106480e+07  1.106480e+07  1.106480e+07  1.106480e+07  1.106480e+07  
            mean    4.792818e+00  1.181388e+00  4.657463e+00  4.855681e+00  1.183553e+00  
            std     8.417202e-01  8.301784e-01  1.117917e+00  7.452962e-01  8.366755e-01  
            min     1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  
            25%     5.000000e+00  1.000000e+00  5.000000e+00  5.000000e+00  1.000000e+00  
            50%     5.000000e+00  1.000000e+00  5.000000e+00  5.000000e+00  1.000000e+00  
            75%     5.000000e+00  1.000000e+00  5.000000e+00  5.000000e+00  1.000000e+00  
            max     5.000000e+00  5.000000e+00  5.000000e+00  5.000000e+00  5.000000e+00  

#### 多值特征的统计：

    word_vec:  count top_20%

    this is feature: appIdAction
    word_vec:  6215 1243
    this is feature: appIdInstall
    word_vec:  64856 12971
    this is feature: interest1
    word_vec:  123 24
    this is feature: interest2
    word_vec:  81 16
    this is feature: interest3
    word_vec:  11 2
    this is feature: interest4
    word_vec:  11 2
    this is feature: interest5
    word_vec:  137 27
    this is feature: kw1
    word_vec:  259909 51981
    this is feature: kw2
    word_vec:  49197 9839
    this is feature: kw3
    word_vec:  11922 2384
    this is feature: topic1
    word_vec:  10001 2000
    this is feature: topic2
    word_vec:  9980 1996
    this is feature: topic3
    word_vec:  5873 1174

#### 选择top20%以后的统计描述信息

    this is feature: interest1
    word_vec:  123 24
    count    1.142004e+07
    mean     8.807826e+00
    std      5.239412e+00
    min      1.000000e+00
    25%      4.000000e+00
    50%      9.000000e+00
    75%      1.300000e+01
    max      2.400000e+01
    Name: interest1, dtype: float64
    this is feature: interest2
    word_vec:  81 16
    count    1.142004e+07
    mean     2.676273e+00
    std      2.391842e+00
    min      1.000000e+00
    25%      1.000000e+00
    50%      2.000000e+00
    75%      4.000000e+00
    max      1.500000e+01
    Name: interest2, dtype: float64
    this is feature: interest5
    word_vec:  137 27
    count    1.142004e+07
    mean     9.560934e+00
    std      6.553343e+00
    min      1.000000e+00
    25%      1.000000e+00
    50%      1.000000e+01
    75%      1.500000e+01
    max      2.600000e+01
    Name: interest5, dtype: float64
    this is feature: kw1
    word_vec:  263311 52662
    count    1.142004e+07
    mean     4.227658e+00
    std      1.328209e+00
    min      1.000000e+00
    25%      4.000000e+00
    50%      5.000000e+00
    75%      5.000000e+00
    max      5.000000e+00
    Name: kw1, dtype: float64
    this is feature: kw2
    word_vec:  49779 9955
    count    1.142004e+07
    mean     4.680158e+00
    std      9.268412e-01
    min      1.000000e+00
    25%      5.000000e+00
    50%      5.000000e+00
    75%      5.000000e+00
    max      5.000000e+00
    Name: kw2, dtype: float64
    this is feature: topic1
    word_vec:  10001 2000
    count    1.142004e+07
    mean     3.636819e+00
    std      1.328687e+00
    min      1.000000e+00
    25%      3.000000e+00
    50%      4.000000e+00
    75%      5.000000e+00
    max      5.000000e+00
    Name: topic1, dtype: float64
    this is feature: topic2
    word_vec:  9983 1996
    count    1.142004e+07
    mean     3.840774e+00
    std      1.245283e+00
    min      1.000000e+00
    25%      3.000000e+00
    50%      4.000000e+00
    75%      5.000000e+00
    max      5.000000e+00
    Name: topic2, dtype: float64
