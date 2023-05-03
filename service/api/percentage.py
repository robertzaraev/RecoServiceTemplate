import pandas as pd


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def get_percentage(user_id: int, reco: list):
    d_val = {}
    d_reco = {}

    item = pd.read_csv('data_original/items.csv')
    inter = pd.read_csv('data_original/interactions.csv')

    for i in item[item['item_id'].isin(
        list(inter[inter['user_id'] == user_id]['item_id']))]['genres']:
        for j in i.replace(" ", "").split(','):
            if j not in d_val.keys():
                d_val[j] = 1
            else:
                d_val[j] += 1
    val = dict(sorted(d_val.items(), reverse=True, key=lambda x: x[1]))

    for i in item[item['item_id'].isin(reco)]['genres']:
        for j in i.replace(" ", "").split(','):
            if j not in d_reco.keys():
                d_reco[j] = 1
            else:
                d_reco[j] += 1
    rec = dict(sorted(d_reco.items(), reverse=True, key=lambda x: x[1]))

    intersections = intersection(list(val.keys()), list(rec.keys())[:10])
    return int(len(intersections) / len(rec.keys()) * 100)
