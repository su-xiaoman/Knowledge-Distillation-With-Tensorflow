import tensorflow as tf
from tensorflow import keras

def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32)/255
    # x = x.astype('float32')/255.0
    y = tf.cast(y,dtype=tf.int32)
    # y = y.astype('int32')
    return x, y


def create_dataset():
    (train_data, train_labels), (test_data,test_labels) = keras.datasets.cifar10.load_data()
    train_labels, test_labels = tf.squeeze(train_labels), tf.squeeze(test_labels)

    BATCH_SIZE = 256
    BUFFER_SIZE = len(train_labels)

    train_db = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_db = train_db.map(preprocess).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    test_db = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    test_db = test_db.map(preprocess).batch(BATCH_SIZE)

    return train_db, test_db
