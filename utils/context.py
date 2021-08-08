import time

class Context:
    def __init__(self,uid):
        self.uid=uid
        self.now = int(time.time())
        self.req_id=str(uid)+"_"+str(self.now)
        self.groups=[]