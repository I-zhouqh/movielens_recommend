import random
from torch.utils.data import Dataset
import pandas as pd
from utils.BFS import BFS

class MovieDataset(Dataset):

    def __init__(self):
        self.bfs=BFS()
        self.rating = pd.read_csv("csvdata/rating.csv")

    def __getitem__(self, index):
        row = self.rating.loc[index]  # uid,gid,score,timestamp

        info = {}
        info['user']=self.bfs.fetch_user_fid(row['uid'])
        info['context'] = self.bfs.fetch_context_fid(row['uid'], row['timestamp'])
        info['group']=self.bfs.fetch_group_fid(row['gid']) # get recent gid. 记住这是一条训练数据，timestamp表示该人评出这个评分的时候的时间。context想要得到的是在这条数据之前看过哪些电影。

        return info, row['score']-1  # 为啥要-1，因为之前的分数是1,2,3,4,5嘛，要以0开始

    def __len__(self):
        return len(self.rating)

class My_dataloader:

    def __init__(self,dataset,batch_size,shuffle=True):
        self.dataset=dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.length=len(dataset)
        self.slices= self.length//self.batch_size

        self.index = [i for i in range(len(self.dataset))]
        if shuffle:
            random.shuffle(self.index)

    def __len__(self):
        return self.slices

    def __iter__(self):
        self.now_slice = 0  # 重置，这样才可以反复使用，要不然for一次就莫得了
        return self

    def __next__(self):
        if self.now_slice >= self.slices:
            raise StopIteration

        result=[]
        for aa in range(self.now_slice*self.batch_size,(self.now_slice+1)*self.batch_size):
            select_index = self.index[aa]
            result.append( self.dataset[select_index] )

        self.now_slice+=1

        return result

if __name__=='__main__':

    dataset=MovieDataset()
    dataloader=My_dataloader(dataset, 32, True)

    for kk in dataloader:
        print(kk)
