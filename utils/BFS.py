import pandas as pd
import bisect


class BFS:

    def __init__(self):
        import os
        # print(os.chdir('../'))
        # print(os.getcwd())
        self.rating = pd.read_csv("csvdata/rating.csv")
        self.user = pd.read_csv("csvdata/user.csv")
        self.item = pd.read_csv("csvdata/item.csv", encoding='gbk')
        self.occupation = pd.read_csv("csvdata/occupation.csv")

    def fetch_user_fid(self,user_id):
        user_info = self.user.query("uid==@user_id")
        result = {}
        result['uid'] = user_id
        result['job'] = self.fetch_job_index(user_info)
        result['sex'] = self.fetch_sex_index(user_info)
        result['age'] = self.fetch_age_index(user_info)
        return result

    def fetch_job_index(self,user_info):
        job = user_info['occupation'].item()
        job_index = self.occupation.query("occupation==@job")['index'].item()
        return job_index

    def fetch_sex_index(self,user_info):
        sex = 1 if user_info['sex'].item() == 'M' else 0
        return sex

    def fetch_age_index(self,user_info):
        age = user_info['age'].item()
        age_index = bisect.bisect([18.5, 28.5, 37, 46], age)  # 一共分成了四类
        return age_index

    def fetch_context_fid(self, user_id, req_time):
        # 记住，一定要选出在这个时刻之前的电影！
        recent_films = \
        self.rating.query("uid==@user_id and timestamp<@req_time").sort_values(by="timestamp", ascending=False)['gid'][
        :3].tolist()
        return {'recent_films': recent_films}

    def fetch_group_fid(self,gid):
        item_info = self.item.query("gid==@gid")
        result = {}
        result['gid'] = gid
        result['pubtime'] = self.fetch_pubtime_index(item_info)
        result['category'] = self.fetch_category_index(item_info)
        return result

    def fetch_pubtime_index(self, item_info):


        pub_time = item_info['pub_time'].item()
        pub_time = int(pub_time.split('-')[2])
        pub_time_index = bisect.bisect([68, 90, 93.5, 96.8], pub_time)
        return pub_time_index

    def fetch_category_index(self, item_info):
        result = []
        for i in range(1, 20):
            value = item_info['type_' + str(i)].item()
            if value == 1:
                result.append(i)
        return result