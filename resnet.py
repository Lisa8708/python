import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, metrics, layers, Sequential, losses, optimizers
import datetime
#
tf.enable_eager_execution()

# ResNet:element-wise addition -> DenseNet:concat
(x, y), (x_test, y_test) = datasets.cifar100.load_data()

# basic block: 2层卷积 + 短接层：identity
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, kernel_size=(3,3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, kernel_size=(3,3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1: # 试输入的大小与卷积后的大小保持一致
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1,1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        out = layers.add([out, identity])
        out = tf.nn.relu(out)

        return out

# resNet 由多个basic block构成
class ResNet(keras.Model):
    def __init__(self, block_list, num_class=100):
        super(ResNet, self).__init__()

        self.stem = Sequential([
            layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2,2), strides=(1,1), padding='same')
        ])
        self.layer1 = self.build_resblock(64,  block_list[0])
        self.layer2 = self.build_resblock(128, block_list[1], stride=2)
        self.layer3 = self.build_resblock(256, block_list[2], stride=2)
        self.layer4 = self.build_resblock(512, block_list[3], stride=2)

        # output: [b, 512, h, w]
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_class)

    def call(self, inputs, training=None):
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x


    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))

        for i in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks

def resnet18():
    return ResNet([2,2,2,2])  # layer_dims:basic_block的数量

def resnet34():
    return ResNet([3,4,6,3])