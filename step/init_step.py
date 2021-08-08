import json
import pandas as pd

class init_step:

    def process(self,ctx):
        init_params(ctx)
        init_groups(ctx)


def init_params(ctx):
    with open('params.json', 'r') as f:
        params = json.load(f)
        print(params)
        ctx.params=params

def init_groups(ctx):
    gids=pd.read_csv('csvdata/item.csv',encoding='gbk')['gid'].tolist()
    ctx.groups=gids