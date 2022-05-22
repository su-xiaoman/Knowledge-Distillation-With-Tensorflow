# 知识蒸馏tensorflow版本基础教程

## 配置及总结

- 数据集为cifar10
- 教师网络TeacherNet可以使用像resnet/transformer等大模型，但是一是实验目的仅仅是为了学习，不需要那么大的模型。二是使用训练好的超大模型太消耗时间，因此仅仅使用自己写的一个类似于vgg的模型，由超过50万的参数。学生网络TeacherNet为了尽可能模拟蒸馏，只有9万参数。
- 按照Yonglong Tian的研究，类似的网络结构的迁移学习基本上只有0-7%的准确率差距，而不同的网络结构有着5-15%的准确率差距

## 结果展示

| 教师网络   | 学生网络   | 正确率(教师=>学生(从0开始训练)) |
| ---------- | ---------- | ------------------------------- |
| TeacherNet | TeacherNet | 76.65%=>70.75%(64.6%)           |

## 模型介绍 

### 教师网络 
- 目前由55万参数组成，由类似于vggnet的组合而成
- 大概20个回合之后：准确率达到了0.7665
### 学生网络
- 学生网络的训练不像教师网络那些是单独进行的，而是基于蒸馏网络生成的
- 学生网络的参数量只有9万参数量，远远少于教师网络的参数量
- $lr=1e-5$
  - 40个epoch:   0.4816
  - 不能像迁移学习一样，使用非常小的学习率，否则训练不动。应该视作是一种新的模型，但是带有额外的约束损失。并且综合各种信息，视作是一种正则化约束似乎更好一点
  - [知识蒸馏视作是一种加速训练方法似乎更好](https://www.bilibili.com/video/BV1G54y1a7hf?spm_id_from=333.337.search-card.all.click)
- $ lr=1e-3，\alpha=0.1 $ 
	- best result: 0.7075
- Student_from_scratch
  - best: acc 64.6%


## 三种损失函数
- hard label: 类似于[1,0,0,0]这样的具有one_hot编码的，soft labels: 由训练过的教师给出，如[0.8,0.05,0.02,0.013]

- 其中一种损失是学生网络的预测和教师网络预测的分布相似性,由于仅仅衡量对于同一随机变量的分布相似性，不使用ground_truth，使用交叉殇是没有道理的。所以定义这种相似性的损失为KLDivergence,它们是基于软标签做的而且伴随着蒸馏温度T, 称为distillation_loss
	
	- ```python
	  distillation_loss = tf.nn.softmax(teachearnet_logits/T)+tf.nn.softmax(studentnet_logits/T)
	  ```
	
- 另外一种损失是学生网络的预测和groundtruth做的，它是基于传统的硬标签的

  - ```python
    student_loss = keras.losses.SparseCategoricalCrossEntropy(from_logits=True)
    ```

- 综合损失由前面两者的加成所做的，其中$\alpha$ 为调节两者的超参数，一般选取小于0.1-0.3从而让主动让分布调节到更接近后者，$ loss = \alpha \times student\_loss + (1-\alpha) \times distillation\_loss $ 

## 训练

### 训练流程

1. 由于这里的教师网络并没有像传统的resnet50那样，有着数千万的参数，因此可以从头开始训练，并且可以将参数保留下来

2. 对于参数量超过千万的超大模型，可以保存权重，多次训练。再然后导入这个权重进行进一步的训练

3. 由于BN/Dropout引入的额外随机性，对于已经训练好的教师网络，在辅助学生模型进行训练时，一律设置training=False

	- ``` python
   teacher_logits = TeacherNet(data,training=False)
   
4. 对比实验中，使用student_from_scratch

   1. tf.keras.models.model.clone_model对于subclass型定义的模型不能用
   2. 使用tensorflow官网介绍的[方法](https://tensorflow.google.cn/api_docs/python/tf/keras/models/clone_model): new_model = model.__class__.from_config(model.get_config())需要自行实现get_config()有点麻烦。因此这里直接使用了不带checkpoint的student_from_scratch = student_model。进行结果对比




## 参考链接
- 论文下载地址[参考链接arxiv](https://arxiv.org/abs/1503.02531)

- tensorflow实现的简单知识蒸馏网络 [参考链接 tencent云](https://cloud.tencent.com/developer/article/1988284)

- 知识蒸馏介绍 [参考链接 devopeida](https://devopedia.org/knowledge-distillation#sample-code)

- 知识蒸馏论文解读[参考链接bilibili](https://www.bilibili.com/video/BV1gS4y1k7vj?spm_id_from=333.337.search-card.all.click)

- Yonglong Tian大佬的案例[参考链接github](https://github.com/HobbitLong/RepDistiller)

- 知识蒸馏损失函数架构 ![参考链接](./reference/arch.png)

  
