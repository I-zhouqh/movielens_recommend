import pandas as pd

class drop_dup_step:

    def process(self,ctx):
        user_id=ctx.uid
        groups=ctx.groups.copy()
        has_read=pd.read_csv("csvdata/rating.csv").query("uid==@user_id")['gid'].tolist()
        ctx.groups=list(set(groups)-set(has_read))