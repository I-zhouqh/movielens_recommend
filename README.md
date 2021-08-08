# 推荐系统尝试(movielens数据集)

## 简介
在推荐部门实习了四个月呢，自己却推荐的完整项目都没有写过，属实还是遗憾，于是来尝试一下！

架构是借鉴了公司的代码，这没得说，主要是多个steps的结构；模型很简单，召回只有三路，有模型的召回只有一路，省略粗排，精排的召回用的是一个deepFM模型（[参考文档](https://zhuanlan.zhihu.com/p/332786045) ），通过这个deepFM模型，对pytorch的embedding怎么搞算是比较熟悉了。

花了零碎的三四天吧，每天三四个小时这样，做出来自己还是蛮有成就感的，记一个todo吧：整一个可视化网页来模拟请求、返回结果的过程（就像字节的sorttools一样），不过有没有时间就不晓得嘞。

总之是一次不错的尝试，周哥，牛逼！开始还担心自己写不出来了，其实还是ok的，嘿嘿。

## 代码结构

### 数据

ml-100k是原始数据，download from [kaggle](https://www.kaggle.com/prajitdatta/movielens-100k-dataset)

csvdata是我手动转换成了csv数据，原始格式不太方便

### serving

参数写在params.json里面这很明白了

main.py是入口，这里定义哪个user去请求，一个请求的生命周期就会走完所有steps，从用户请求开始，到返回推荐结果结束。

1. init_step: 读进所有视频，读进配置文件
2. drop_dup_step: 去重
3. recall_step: 召回。共高热召回、兴趣召回（用户最近看过的五个电影中，出现最频繁的种类是哪一类，于是这路召回召回出该类的数条视频）、模型召回（group embedding内积user embedding，然后加上group bias）。还有写了个蛇形排序，感觉比较tricky，比较有意思，hhh。
4. predict_step: 精排。很简单，调精排模型即可。难的地方都在训练时做了
5. info_show_step: 展示结果。将具体的推荐的结果展示出来。 这里仅仅是print，后期考虑放在网站上。

### train

serving仅仅是在线部分。当然还有训模型的离线部分。在model文件夹下，

model.py是viking召回模型和deepFM模型;Dataset.py就是pytorch常见的dataset类了，里面还有我自己写的dataloader（因为pytorch的dataloader必须要求Dataset中getitem返回的是tensor，因为这个问题的特殊性，我的Dataset的getitem无法返回tensor，否则太麻烦了，所以我只能放弃pytorch的dataloader，转而自己写loader，其实很简单，注意iter方法和next方法就可以了，受益匪浅。

train_predict_model.py与train_recall_model.py就分别是训练两个模型的文件了，分别运行即可，日志和表现最好的模型（在验证集上）都会保存到data文件中，这个保存的模型会在serving时被调用

### 运行

python main.py --uid 3