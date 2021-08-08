import pandas as pd
import torch
from model.model import DeepFM
from utils.BFS import BFS

class predict_step:

    def process(self,ctx):
        do_predict(ctx)


def do_predict(ctx):
    all_info=get_all_info(ctx)  # list of dict
    model=load_model(ctx.params['predict']['model_path'])
    logits = model(all_info)
    groups = sort_results(ctx.groups, logits)  #作排序
    predict_count=ctx.params['predict']['count']
    groups = groups[:predict_count]
    ctx.groups=groups


def sort_results(groups, logits):
    # 既要看判断出属于哪个类别，也要看属于这个类别的概率是不是够大

    max_index = torch.argmax(logits, dim=1)
    max_value = torch.max(logits, dim=1)[0].detach().numpy()

    df = pd.DataFrame()
    df['max_index']=max_index
    df['max_value']=max_value
    df['groups']=groups
    df.sort_values(by=['max_index','max_value'],ascending=[False,False])  #先看类别，类别一样看概率
    return df['groups'].tolist()

def load_model(path):
    model = DeepFM()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def get_all_info(ctx):
    all_info=[]

    bfs = BFS()  #特征抽取服务

    user=bfs.fetch_user_fid(ctx.uid)
    context=bfs.fetch_context_fid(ctx.uid, ctx.now ) # get recent gid

    info={}
    info['user']= user
    info['context']= context

    for group in ctx.groups.copy():
        info['group'] = bfs.fetch_group_fid(group)
        all_info.append(info.copy())

    return all_info


