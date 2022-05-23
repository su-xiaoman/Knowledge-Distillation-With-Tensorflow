import tensorflow as tf
from tensorflow import keras
from keras import layers


class TeacherNet(tf.keras.Model):
    def __init__(self):

        super(TeacherNet, self).__init__()
        #unit = conv1 + bn + relu
        self.conv1 = layers.Conv2D(32, (3,3),padding="same",activation="relu")
        self.conv2 = layers.Conv2D(32, (3,3),padding="same",activation="relu")
        self.maxpool1 = layers.MaxPooling2D((2,2))
        self.dropout1 = layers.Dropout(0.5)

        self.conv3 = layers.Conv2D(64,(3,3),padding="same",activation="relu")
        self.conv4 = layers.Conv2D(64,(3,3),padding="same",activation="relu")
        self.maxpool2 = layers.MaxPooling2D((2,2))
        self.dropout2 = layers.Dropout(0.5)

        self.conv5 = layers.Conv2D(128,(3,3),padding="same",activation="relu")
        self.conv6 = layers.Conv2D(128,(3,3),padding="same",activation="relu")
        self.maxpool3 = layers.MaxPooling2D((2,2))
        self.dropout3 = layers.Dropout(0.5)

        self.flatten = layers.Flatten()
        self.avgpool = layers.AveragePooling2D()

        self.d1 = layers.Dense(128, activation="relu")
        self.dropout4 = layers.Dropout(0.5)
        self.d2 = layers.Dense(10, name='logits')

    def call(self, input, training=None):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.avgpool(x)

        x = self.d1(x)
        x = self.dropout4(x)
        out = self.d2(x) #这里的out输出的logits而非softmax

        return out

if __name__ == '__main__':
    model = TeacherNet()
    model.build(input_shape=[None,32,32,3])
    model.summary()
    # model

    # model = keras.compile()
    # model.fit()
    # EPOCH, train_db, model, optimizer
    # train(EPOCH=20,train_db=)