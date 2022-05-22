import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import time
from tqdm import tqdm

# @tf.function
def train_step(train_db, model=None ):

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    for step, (X, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            logits = model(X, training=True)
            y_onehot = tf.one_hot(y, depth=10)
            loss = criterion(y_onehot, logits)
            # print(f"no mean, loss dim: {loss}, {loss.shape}")
            loss = tf.reduce_mean(loss)
            # print(f"with mean, loss dim: {loss}, {loss.shape}")

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(f"step:{step},loss:{float(loss)}")




def test_step(test_db,model=None,total_num=0, total_correct=0):
    #对于测试集来说，只要一次测评结果就行
    for x, y in test_db:
        # with optimizers
        logits = model(x, training=False)
        # out = tf.reshape(out,[-1,512])
        # logits = fc_net(out)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_num += x.shape[0]
        total_correct += int(correct)
    acc = total_correct / total_num
    print('acc', acc)