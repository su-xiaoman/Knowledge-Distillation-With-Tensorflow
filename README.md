# 知识蒸馏tensorflow版本基础教程

## 配置及总结

- 数据集为cifar10
- 框架为:tensorflow 2.8.0 
- Python: 3.9.11
- 教师网络TeacherNet可以使用像resnet/transformer等大模型，但是一是实验目的仅仅是为了学习，不需要那么大的模型。二是使用训练好的超大模型太消耗时间，因此仅仅使用自己写的一个类似于vgg的模型，由超过50万的参数。学生网络TeacherNet为了尽可能模拟蒸馏，只有9万参数。
- 按照Yonglong Tian的研究，类似的网络结构的迁移学习基本上只有0-7%的准确率差距，而不同的网络结构有着5-15%的准确率差距
- **此部分列举尚未完成的工作** 
  - 从实验效果上来说，并没有显著的说服力，这可能是因为对于cifar10数据集来说，大多数简单的模型就足以保证70%的准确率，这使得迁移学习的提升并未展示地非常明显。在未来可以使用像resnet/transformer模型来做为教师模型，然后设计一个简单的模型，最后使用from scratch和kd两种方法来对比效果。另外由于本实验所采用的tensorflow框架缺乏像pytorch.timm那样的现有模型与权重，其keras.applications只有极少的模型，也不便于进一步实验，因此没有进行进一步讨论。
  - 本实验同样没有添加BN更多正则化效果
  - 没有使用集成学习的方案以测试知识蒸馏对于集成方案的学习效果会怎样


## 结果展示

| 教师网络   | 学生网络   | 正确率(教师=>学生(从0开始训练)) |
| ---------- | ---------- | ------------------------------- |
| TeacherNet | TeacherNet | 76.65%=>71.98% (71.6%)          |

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
### 论文要点解读
> When the correct labels are known for all or some of the transfer set, this method can be significantly improved by also training the distilled model to produce the correct labels.

- 这里说对于迁移集来说，如果能够知道部分或全部的准确的标签的话，其方法就能够借助同时训练学生模型去产生正确的标签从而得到模型的改善
> One way to do this is to use the correct labels to modify the soft targets, but we found that a better way is to simply use a weighted average of two different objective functions. The first objective function is the cross entropy with the soft targets and this cross entropy is computed using the same high temperature in the softmax of the distilled model as was used for generating the soft targets from the cumbersome model. 
- 但是一种更常用的方法是使用两个目标函数(也就是损失函数)的一种加权平均，其中一种损失函数是基于软标签的交叉熵，这种交叉熵的计算是使用和生成这种软标签的大模型一样的温度T

```python
我们记为： distillation_loss = crossentropy(tf.nn.softmax(teacher_logit/T),tf.nn.softmax(student_logit/T))
但是教师网络给予的标签未必是完全正确的，因此我们使用交叉熵的一个变种KL散度(H(P,Q)=H(P)+KL(P||Q)),由于我们缺乏足够可信的H(P),即教师网络的知识未必完全正确，但是其给出的预测分布所代表的知识却足够可信。因此这里的crossentropy=>KLDivergence
```

> The second objective function is the cross entropy with the correct labels. This is computed using exactly the same logits in softmax of the distilled model but at a temperature of 1. We found that the best results were generally obtained by using a condiderably lower weight on the second objective function. Since the magnitudes of the gradients produced by the soft targets scale as 1/T^2 it is important to multiply them by T^2 when using both hard and soft targets. This ensures that the relative contributions of the hard and soft targets remain roughly unchanged if the temperature used for distillation is changed while experimenting with meta-parameters. 

- 这里的第二个损失函数是student_loss, 它是基于正确的标签的交叉熵，通过计算使用精确的得分(设置T=1),并且最好的结果出现在设定相对更低的权重。作者认为被软标签（distillation_loss）所产生的梯度的量级大概会是$1/T^2$ ，所以需要给没有除以T的student_loss乘上相对更低的系数以补偿。但是从实现来看，这个$1/T^2$ 很难度量，因此我们仅仅通过实验给予最好的权重系数$\alpha$，比如0.1/0.3感觉实验效果都还行


### 训练流程

1. 由于这里的教师网络并没有像传统的resnet50那样，有着数千万的参数，因此可以从头开始训练，并且可以将参数保留下来

2. 对于参数量超过千万的超大模型，可以保存权重，多次训练。再然后导入这个权重进行进一步的训练

3. 由于BN/Dropout引入的额外随机性，对于已经训练好的教师网络，在辅助学生模型进行训练时，一律设置training=False

  - ``` python
    teacher_logits = TeacherNet(data,training=False)

4. 对比实验中，使用student_from_scratch

   1. tf.keras.models.model.clone_model对于subclass型定义的模型不能用
   2. 使用tensorflow官网介绍的[方法](https://tensorflow.google.cn/api_docs/python/tf/keras/models/clone_model): new_model = model.__class__.from_config(model.get_config())需要自行实现get_config()有点麻烦。因此这里直接使用了不带checkpoint的student_from_scratch = student_model。进行结果对比

5. 当我们设定最终的loss完完全全由软标签的教师网络给出而没有任何的ground_true时(代码如下所示)，即我们的学生网络几乎不知道正确的标签是什么的情况下，依然能够凭借教师网络所传授的知识给予最终的判断，在12个epoch后达到了0.6368的准确度

   1. ```  python
      teacher_logits = teacher_model(data, training=False)
      studnet_logits = student_model(data, training=True)
      loss = distillation_loss(teacher_logits,studnet_logits)
      ```

6. 本实验没有对论文中的专才通才模型做出实验，因为一方面很难找到并且在短时间内训练出那样的一个模型，二是同样缺乏并行条件来实验论文中所提出的数据并行和模型并行，本文的目标在于学习迁移学习的知识并且应用于未来的研究中。

## 参考链接
- 论文下载地址[参考链接arxiv](https://arxiv.org/abs/1503.02531)

- tensorflow实现的简单知识蒸馏网络 [参考链接 tencent云](https://cloud.tencent.com/developer/article/1988284)

- 知识蒸馏介绍 [参考链接 devopeida](https://devopedia.org/knowledge-distillation#sample-code)

- 知识蒸馏论文解读[参考链接bilibili](https://www.bilibili.com/video/BV1gS4y1k7vj?spm_id_from=333.337.search-card.all.click)

- Yonglong Tian大佬的案例[参考链接github](https://github.com/HobbitLong/RepDistiller)

- 知识蒸馏损失函数架构 ![参考链接](./reference/arch.png)

  
