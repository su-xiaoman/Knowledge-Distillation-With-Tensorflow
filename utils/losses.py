import tensorflow as tf
from tensorflow import keras
from keras import losses
import numpy as np

# 蒸馏温度T测试
# def distillation_softmax(logits, temperature=3):
#     # 这里的temperatures为蒸馏温度，logits为经过网络后的得分向量
#     logits = logits / temperature
# #     return np.exp(logits)/np.sum(np.exp(logits)) #这个实质上本身就是一个softmax
#     return tf.nn.softmax(logits/temperature)

# T=3: 计算学生网络和教师网络对于同一数据集预测结果的分布差异
def distillation_loss_fn(studentnet_logits, teachernet_logits):
    criterion = tf.keras.losses.KLDivergence()
    loss = criterion(tf.nn.softmax(teachernet_logits/5, axis=1),
                     tf.nn.softmax(studentnet_logits/5, axis=1))
    return loss

# T=1: student_pred,ground_truth
def student_loss_fn(student_logit, target):

    criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss = criterion(target, student_logit)

    return loss

if __name__ == '__main__':
    # teacher_logit = tf.constant([1,2,5,2])
    teacher_logit = tf.constant([[0.9, 0.6, 0.2, 0.1],[0.9, 0.6, 0.2, 0.1]])
    # student_logit = tf.constant([[0.1,0.2,0.5,0.2],[0.1,0.2,0.5,0.2]])
    # print(distillation_softmax(teacher_logit), tf.reduce_sum(distillation_softmax(teacher_logit)))
    # print(distillation_softmax(student_logit))
    # print(distillation_loss_fn(teacher_logit,student_logit))