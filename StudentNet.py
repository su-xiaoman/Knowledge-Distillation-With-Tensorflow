import tensorflow as tf
from tensorflow import keras
from keras import layers

class StudentNet(keras.Model):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), padding="same",activation="relu",name="this_is_conv1")
        self.conv2 = layers.Conv2D(32, (3, 3),padding="same",activation="relu")

        self.conv3 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")
        self.conv4 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")

        self.maxpool1 = layers.MaxPooling2D((2, 2))
        self.dropout1 = layers.Dropout(0.5, name="this_is_dropout1")
        self.maxpool2 = layers.MaxPooling2D((2, 2))
        self.dropout2 = layers.Dropout(0.5)

        self.avgpool = layers.GlobalAveragePooling2D()
        # self.flatten = layers.Flatten(name="my_name_is_flatten")
        self.d1 = layers.Dense(40,activation="relu")
        # self.dropout2 = layers.Dropout(0.5)

        self.d2 = layers.Dense(10, name='logits')
        # self.activ1 = layers.Activation('softmax')

    def call(self, input, training=None):
        # module1
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)


        x = self.avgpool(x)
        print(f"x.shape:{x.shape}") #256,8,8,32
        # x = self.d1(x)
        # x = self.dropout2(x)
        output = self.d2(x)

        return output

if __name__ == '__main__':
    model = StudentNet()
    model.build(input_shape=[None,32,32,3])
    model.summary()