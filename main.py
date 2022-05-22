import os
# os.environ['CUDA_VISIBLE_DEVICE'] = ''

import tensorflow as tf
from tensorflow import keras
from TeacherNet import TeacherNet
from StudentNet import StudentNet
from utils.train import train_step
from utils.train import test_step
from utils.create_dataset import create_dataset
from tqdm import tqdm
from utils.losses import distillation_loss_fn, student_loss_fn

if __name__ == '__main__':
    #  define dataset
    (train_db, test_db) = create_dataset()
    # define model
    teacher_model = TeacherNet()
    student_model = StudentNet()

    # define parameters
    optimizer = tf.keras.optimizers.Adam(1e-3)
    alpha = 0.2
    EPOCHS = 12
    teacher_checkpoint_save_path = "./checkpoints/teacher_checkpoint.ckpt"
    student_checkpoint_save_path = "./checkpoints/student_checkpoint.ckpt"


    # train from scratch
    # for epoch in range(EPOCHS):
    #     train_step(train_db, teacher_model)
    #     test_step(test_db, teacher_model)
    # teacher_model.save_weights(student_checkpoint_save_path)

    if os.path.exists(teacher_checkpoint_save_path + '.index'):
        teacher_model.load_weights(teacher_checkpoint_save_path)
        print("teacher checkpoint load success!")

    if os.path.exists(student_checkpoint_save_path + '.index'):
        student_model.load_weights(student_checkpoint_save_path)
        print("student checkpoint load success!")


    for epoch in range(EPOCHS):
        # 本实验采用的cifar10由于数据量过小,使用epoch可能并没有那么多意义，因此选用可以观察进度的tqdm
        # for step, (data, target) in enumerate(train_db):
        for data, target in tqdm(train_db):
            #前向传播
            with tf.GradientTape() as tape:
                teacher_logits = teacher_model(data, training=False)
                student_logits = student_model(data, training=True) #float32,(256,10)
 
                target = tf.one_hot(target, depth=10)

                student_loss = student_loss_fn(student_logits, target)
                distillation_loss = distillation_loss_fn(student_logits, teacher_logits)

                loss = alpha * student_loss + (1-alpha) * distillation_loss
            # 反向传播
            grads = tape.gradient(loss, student_model.trainable_variables)
            #梯度更新
            optimizer.apply_gradients(zip(grads, student_model.trainable_variables))


        total_correct = 0
        total_num = 0
        #因为target同样没有经过ont_hot,所以直接是一个相应的数
        for data, target in tqdm(test_db):
            logits = student_model(data, training=False)

            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1) #罗列出其相应的下标,也就是相应的类别
            pred = tf.cast(pred, dtype=tf.int32)

            #torch version
            # pred = prob.max(1).indices
            correct = tf.equal(pred, target)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            #对于整个batch来说，可以把所有的加起来
            total_correct += int(correct)
            total_num += data.shape[0]

        acc = total_correct / total_num
        print(f"{epoch+1},test acc: {acc}")

    if os.path.exists('./checkpoints'):
        student_model.save_weights(student_checkpoint_save_path)
        print("student model checkpoint saved")


