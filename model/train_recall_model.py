from torch import nn
from Dataset import MovieDataset,My_dataloader
from model import Recall
import torch.optim as optim
import torch.utils.data as Data
import time, datetime
from tqdm import tqdm
import numpy as np
import torch



batch_size=32
train_prob=0.95
torch.manual_seed(4396)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


dataset = MovieDataset()
train_data, eval_data = torch.utils.data.random_split(dataset, [int(len(dataset)*train_prob), int(len(dataset)*(1-train_prob))])

train_dataloader = My_dataloader(train_data, batch_size=batch_size, shuffle=True)
eval_dataloader = My_dataloader(eval_data, batch_size = batch_size, shuffle=True)

model=Recall()
model.to(device)


loss_function=nn.MSELoss(reduction='mean')   # 第一个类别（评分1）比较少，加大一下权重，而且它确实影响恶劣一些
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

# 打印模型参数
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
print(get_parameter_number(model))

def write_log(w):
    file_name = 'model/data/data' +"_{}.log".format("recall")
    t0 = datetime.datetime.now().strftime('%H:%M:%S')
    info = "{} : {}".format(t0, w)
    print(info)
    with open(file_name, 'a') as f:
        f.write(info + '\n')



def train_and_eval(model, train_loader, eval_dataloader, epochs, device):
    best_loss = np.inf
    for _ in range(epochs):
        """训练部分"""
        model.train()
        print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        write_log('Epoch: {}'.format(_ + 1))
        train_loss_sum = 0.0
        start_time = time.time()

        #print(train_loader)

        for idx, batch in tqdm(enumerate(train_loader)):
            infos= [ item[0] for item in batch ]  #list of dict, 输入model会自己变成tensor，这里不用管
            labels = [item[1] for item in batch]
            labels=torch.tensor(labels, dtype=torch.float)  # 用MSE, label应该是float的

            logits = model(infos)
            logits = torch.squeeze(logits)   #压到1维

            loss=loss_function(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            if (idx + 1) % 5 == 0 or (idx + 1) == len(train_loader):
                write_log("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                    _ + 1, idx + 1, len(train_loader), train_loss_sum / (idx + 1), time.time() - start_time))
        scheduler.step()
        """推断部分"""
        model.eval()
        write_log("************* now eval *****************")

        loss_array = []
        with torch.no_grad():
            #valid_labels, valid_preds = [], []
            for idx, batch in tqdm(enumerate(eval_dataloader)):
                infos = [item[0] for item in batch]  # list of dict, 输入model会自己变成tensor，这里不用管
                labels = [item[1] for item in batch]
                labels = torch.tensor(labels, dtype=torch.float)

                logits = model(infos)
                logits = torch.squeeze(logits)  # 压到1维

                loss = loss_function(logits, labels)
                loss_array.append(loss.cpu().item())

        cur_loss = np.mean(loss_array)

        if cur_loss < best_loss:
            best_loss = cur_loss
            torch.save(model.state_dict(), "model/data/recall_best.pth")
        write_log('Current loss: %.6f, Best loss: %.6f\n' % (cur_loss, best_loss))


train_and_eval(model, train_dataloader, eval_dataloader, 15, device)

