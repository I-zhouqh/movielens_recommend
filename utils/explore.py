import pandas as pd

# create hot.csv, for hot recall.

df=pd.read_csv("../movielens/csvdata/rating.csv")
tmp = df.groupby('gid').agg({'score':'mean','uid':'count'}).reset_index().rename(columns={'score':'avg_score','uid':'watch_times'})
dd=tmp.query("watch_times>=10").sort_values(by="avg_score",ascending=False)
dd.to_csv("csvdata/hot.csv",index=False)
print("update hot movie successfully")