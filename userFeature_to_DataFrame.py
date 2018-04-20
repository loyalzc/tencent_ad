# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/17 15:44
@Function:
"""

from csv import DictWriter

with open('input/userFeature.csv', 'w') as out_f:
    headers = ['uid', 'age', 'gender', 'marriageStatus', 'education', 'consumptionAbility', 'LBS', 'interest1', 'interest2',
               'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3',  'topic1', 'topic2', 'topic3', 'appIdInstall',
               'appIdAction', 'ct', 'os', 'carrier', 'house']
    writer = DictWriter(out_f, fieldnames=headers, lineterminator='\n')
    writer.writeheader()

    in_f = open('input/userFeature.data', 'r')
    for t, line in enumerate(in_f, start=1):
        line = line.replace('\n', '').split('|')
        userFeature_dict = {}
        for each in line:
            each_list = each.split(' ')
            userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
        writer.writerow(userFeature_dict)
        if t % 100000 == 0:
            print(t)
    in_f.close()

