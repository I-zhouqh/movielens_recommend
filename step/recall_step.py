import random
import pandas as pd
import numpy as np
from utils.BFS import BFS
import torch
from model.model import Recall

class recall_step:

    def process(self,ctx):
        do_recall(ctx)


def do_recall(ctx):
    all_count=ctx.params['recall']['count']
    recall=ctx.params['recall']['strategy'].copy()

    for method in recall.keys():
        value = recall[method]

        if method=="rating":
            groups=do_recall_viking(ctx,all_count)
            value['groups']=groups
            value['now_in'] = 0
        elif method=="hot":
            groups=do_recall_hot(ctx,all_count)
            value['groups']=groups
            value['now_in'] = 0
        elif method=="recent_interest":
            groups=do_recall_interest(ctx,all_count)
            value['groups']=groups
            value['now_in']=0
        else:
            print("not find", method)

    groups=snake_merge(ctx,recall,all_count)
    ctx.groups=groups

def snake_merge(ctx,recall,all_count):
    ## snake merge
    result=[]
    all_weight=sum(v['weight']  for k,v in recall.items())
    methods=list(recall.keys())
    while len(methods)>0 and len(result)<=all_count:
        for method in methods:
            groups=recall[method]['groups']
            if len(groups)==0: #删除掉这一路召回
                methods.remove(method)
                continue
            if recall[method]['now_in']*all_weight <= recall[method]['weight']*len(result) :
                in_group=groups.pop(0)
                while in_group in result and len(groups) > 0:
                    in_group = groups.pop(0)
                if len(groups)==0:
                    methods.remove(method)
                    continue
                result.append((in_group,method))
                recall[method]['now_in']+=1

        methods.reverse() # 蛇形排序，所以翻转过来

    ctx.recall_reasons=result

    groups = list(map(lambda x:x[0],result))
    return groups

def do_recall_interest(ctx,limit):
    user_id=ctx.uid
    recent_films = pd.read_csv("csvdata/rating.csv").query("uid==@user_id").sort_values(by="timestamp", ascending=False)['gid'][:5]
    items = pd.read_csv("csvdata/item.csv",encoding='gbk').merge(recent_films,on='gid')

    type_col = [col for col in items.columns if col.startswith('type_')]
    items=items[type_col]
    tmp = items.sum().tolist()
    interest=random.choice(np.where(tmp==np.max(tmp))[0])   #在最多的category里面随机挑一个
    interest_type="type_"+str(interest+1)
    items = pd.read_csv("csvdata/item.csv", encoding='gbk')
    interest_movies = items.loc[items[interest_type] == 1]['gid'].tolist()

    results=[]
    count=0
    for movie in interest_movies:
        if count>limit:
            break
        if movie in ctx.groups:
            results.append(movie)
            count+=1
    return results

def do_recall_hot(ctx, limit):
    results=[]
    hot_movies=pd.read_csv("csvdata/hot.csv")['gid'].tolist()
    count=0
    for hot_movie in hot_movies:
        if count>limit:
            break
        if hot_movie in ctx.groups:
            results.append(hot_movie)
            count+=1
    return results

def load_model(path):
    model = Recall()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def do_recall_viking(ctx, limit):

    all_info=get_all_info(ctx)  # list of dict
    model = load_model(ctx.params['recall']['model_path'])
    pred_ratings = model(all_info)
    groups = sort_results(ctx.groups.copy(), pred_ratings)  #作排序
    groups = groups[:limit]
    return groups

def sort_results(groups, pred):
    pred = torch.squeeze(pred).detach().numpy().tolist()
    tmp = sorted(list(zip(groups, pred)), key=lambda x: x[1], reverse=True)
    tmp = list(map(lambda x: x[0], tmp))
    return tmp

def get_all_info(ctx):
    all_info=[]

    bfs = BFS()
    user = bfs.fetch_user_fid(ctx.uid)

    info={}
    info['user']= user

    for group in ctx.groups.copy():
        info['group'] = bfs.fetch_group_fid(group)
        all_info.append(info.copy())
    return all_info