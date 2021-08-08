import json
import pandas as pd
from utils import BFS

class info_show_step:

    def process(self,ctx):
        show_results(ctx.groups.copy())

def show_results(groups):
    item = pd.read_csv("csvdata/item.csv",encoding='gbk')
    genre = pd.read_csv("csvdata/genre.csv")['category'].tolist()
    print("\n\n")
    for group in groups:
        info = item.query("gid==@group")
        movie_name = info['movie_name'].item()
        url = info['url'].item()
        category=[]
        for type in range(19):
            if info['type_'+ str(type + 1)].item()==1:
                category.append(genre[type])

        print("*******************************************\n")
        print(f"recommend movie {movie_name}, id {group}, url:{url}, category:{category}\n")

if __name__=='__main__':
    show_results([127,279])



