# main refer: https://zhuanlan.zhihu.com/p/332786045
# another refer: https://zhuanlan.zhihu.com/p/84526966
from torch import nn
import torch
import random


class Recall(nn.Module):
    def __init__(self, emb_size=4):
        super().__init__()

        user_category_num = [944, 22, 2, 5]  # uid, job ,sex, age
        group_category_num = [1683, 5, 20] # gid, pubtime, category. 多个category随机挑一个

        # first order

        ## 无需user bias，因为它不影响serving
        self.fm_1st_order_group_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in group_category_num])  # 类别特征的一阶表示

        # second order

        self.fm_2nd_order_user_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in user_category_num])  # 类别特征的二阶表示

        self.fm_2nd_order_group_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in group_category_num])  # 类别特征的二阶表示

        self.sigmoid=nn.Sigmoid()

    def forward(self, infos):
        group_1st, user_2nd, group_2nd = self.get_embedding(infos)
        logits = group_1st + torch.sum( user_2nd*group_2nd, dim = 1, keepdim=True)   # embedding作内积，然后加上bias
        return 5*self.sigmoid(logits)  # 0-5之间.  label是0,1,2,3,4


    def get_embedding(self,infos):
        def process_one(info):

            # 相比predict, recall的feature少了recent_films, category也只有一个


            category_chosen = random.choice(info['group']['category']) #有多个的话，随机选一个

            # first order

            group_feature_list =[ info['group']['gid'], info['group']['pubtime'], category_chosen ]
            fm_1st_group_embedding = [emb(torch.tensor(group_feature_list[i])) for i, emb in
                                enumerate(self.fm_1st_order_group_emb)]
            fm_1st_group_embedding = torch.sum(torch.cat(fm_1st_group_embedding))
            # second order

            ## user
            # uid, job ,sex, age
            user_feature_list =[ info['user']['uid'], info['user']['job'], info['user']['sex'] ,info['user']['age'] ]
            fm_2nd_user_embedding = [emb(torch.tensor(user_feature_list[i])) for i, emb in
                                enumerate(self.fm_2nd_order_user_emb)]
            fm_2nd_user_embedding=torch.mean(torch.stack(fm_2nd_user_embedding),dim=0)

            ## group
            group_feature_list =[ info['group']['gid'], info['group']['pubtime'], category_chosen  ]
            fm_2nd_group_embedding = [emb(torch.tensor(group_feature_list[i])) for i, emb in
                                enumerate(self.fm_2nd_order_group_emb)]
            fm_2nd_group_embedding=torch.mean(torch.stack( fm_2nd_group_embedding),dim=0)

            return  fm_1st_group_embedding, fm_2nd_user_embedding, fm_2nd_group_embedding

        result=list(map(process_one, infos))
        group_1st = torch.stack([item[0] for item in result])
        group_1st = torch.unsqueeze(group_1st,1)
        user_2nd = torch.stack([item[1] for item in result])
        group_2nd = torch.stack([item[2] for item in result])

        return group_1st, user_2nd, group_2nd




class DeepFM(nn.Module):
    def __init__(self, emb_size=4,
                 hid_dims=[256, 128, 32], num_classes=5, dropout=[0.2, 0.2, 0.2]):

        super().__init__()
        user_num=944
        group_num=1683

        """FM部分"""
        # job ,sex ,age ,pubtime
        feature_list_cat_num = [22, 2, 5, 5]

        # bias
        ## category就没要，因为是个list，bias不好表示
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in [user_num, group_num] + feature_list_cat_num])  # 类别特征的一阶表示

        # embedding
        self.user_embedding = nn.Embedding(user_num, emb_size)
        self.group_embedding = nn.Embedding(group_num, emb_size, padding_idx=0) # recent_gid中可能有null值，于是规定一个padding_idx
        self.category_embedding = nn.Embedding(20, emb_size)
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in feature_list_cat_num])  # 类别特征的二阶表示

        """DNN部分"""
        self.all_dims = [32] + hid_dims
        #self.dense_linear = nn.Linear(self.nume_fea_size, self.cate_fea_size * emb_size)  # 数值特征的维度变换到FM输出维度一致

        # for DNN
        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i - 1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i - 1]))

        """output部分"""
        # for output
        self.dnn_linear = nn.Linear(hid_dims[-1]+2, num_classes)
        self.softmax=nn.Softmax(dim=1)


    def get_second_order(self,infos):
        def process_one(info):
            # second order
            uid = self.user_embedding(torch.tensor(info['user']['uid']))
            gid = self.group_embedding(torch.tensor(info['group']['gid']))

            if len(info['context']['recent_films']) == 0:
                recent_gid = self.group_embedding(torch.tensor(0))   # 如果没有，直接去取0的embedding（在padding_index中有了保证）
            else:
                recent_gid = torch.sum(
                    torch.stack([self.group_embedding(torch.tensor(g)) for g in info['context']['recent_films']]),
                    dim=0)

            category = torch.mean(
                torch.stack([self.category_embedding(torch.tensor(g)) for g in info['group']['category']]), dim=0)
            # 除了uid和gid和category以外的其他feature
            feature_list = [info['user']['job'], info['user']['sex'], info['user']['age'], info['group']['pubtime']]
            #print(feature_list)

            fm_2nd_order_res = [emb(torch.tensor(feature_list[i])) for i, emb in
                                enumerate(self.fm_2nd_order_sparse_emb)]

            fm_2nd_order_res.append(uid)
            fm_2nd_order_res.append(gid)
            fm_2nd_order_res.append(recent_gid)
            fm_2nd_order_res.append(category)
            fm_2nd_order_res = torch.stack(fm_2nd_order_res)  # size 8*4
            return  fm_2nd_order_res

        result=list(map(process_one, infos))
        result=torch.stack(result)
        return result

    def get_first_order(self,infos):
        def process_one(info):
            # first order
            feature_list = [info['user']['uid'], info['group']['gid'], info['user']['job'], info['user']['sex'], info['user']['age'], info['group']['pubtime']]

            fm_1st_sparse_res = [  emb(torch.tensor( feature_list[i]  ) )
                                 for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
            fm_1st_sparse_res = torch.cat(fm_1st_sparse_res)
            fm_1st_sparse_res = torch.sum(fm_1st_sparse_res)
            return fm_1st_sparse_res

        result=list(map(process_one, infos))
        result=torch.stack(result)
        result=torch.unsqueeze(result, 1)
        return result



    def forward(self, infos):
        """
        X_sparse: 类别型特征输入  [bs, cate_fea_size]
        X_dense: 数值型特征输入（可能没有）  [bs, dense_fea_size]
        """

        """FM部分"""

        fm_1st_sparse_res = self.get_first_order(infos) # bs*1
        fm_2nd_concat_1d = self.get_second_order(infos) # bs*feature_num*emb_size

        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, emb_size]
        square_sum_embed = sum_embed * sum_embed  # [bs, emb_size]
        # 先平方再求和
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, n, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, emb_size]
        # 相减除以2
        sub = square_sum_embed - sum_square_embed
        sub = sub * 0.5  # [bs, emb_size]

        fm_2nd_part = torch.sum(sub, 1, keepdim=True)  # [bs, 1]

        """DNN部分"""
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)  # [bs, n * emb_size]

        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)

        dnn_out = torch.cat([dnn_out,fm_1st_sparse_res,fm_2nd_part],dim=1)  # 拼在一起，从32维变成34维
        dnn_out = self.dnn_linear(dnn_out)  # [bs, 5]
        #logit = self.softmax(dnn_out)  # [bs,5]

        # 都不用softmax了，在损失函数里自己会做了。
        return dnn_out


def test_predict():
    a=DeepFM()

    infos=[]
    infos.append({'user': {'uid': 1, 'job': 20, 'sex': 1, 'age': 1}, 'context': {'recent_films': [74, 102, 5]}, 'group': {'gid': 966, 'pubtime': 0, 'category': [12,15]}})
    infos.append({'user': {'uid': 13, 'job': 20, 'sex': 1, 'age': 1}, 'context': {'recent_films': [74, 102, 5]}, 'group': {'gid': 966, 'pubtime': 0, 'category': [12,15]}})
    a.forward(infos)

def test_recall():
    a=Recall()

    infos=[]
    infos.append({'user': {'uid': 1, 'job': 20, 'sex': 1, 'age': 1}, 'context': {'recent_films': [74, 102, 5]}, 'group': {'gid': 966, 'pubtime': 0, 'category': [12,15]}})
    infos.append({'user': {'uid': 13, 'job': 20, 'sex': 1, 'age': 1}, 'context': {'recent_films': [74, 102, 5]}, 'group': {'gid': 966, 'pubtime': 0, 'category': [12,15]}})
    infos.append({'user': {'uid': 13, 'job': 20, 'sex': 1, 'age': 1}, 'context': {'recent_films': [74, 102, 5]}, 'group': {'gid': 966, 'pubtime': 0, 'category': [12,15]}})

    a.forward(infos)

#if name=='__'
if __name__=='__name__':
    test_recall()