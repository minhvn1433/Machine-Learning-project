import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
import parameter

VGG16_dp_rate_01 = 0.1

class Model():
    def __init__(self, num_classes):
        self.PT_model = VGG16(
            input_shape=(parameter.IMAGE_SIZE,parameter.IMAGE_SIZE,3),
            include_top=False,
            weights='imagenet'
        )
        for layer in self.PT_model.layers:
            layer.trainable = False
        
        x = keras.layers.Flatten()(self.PT_model.output)
        x = keras.layers.Dense(num_classes*4, activation='relu')(x)
        x = keras.layers.Dropout(VGG16_dp_rate_01)(x)
        x = keras.layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = keras.models.Model(self.PT_model.input, x)

        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.legacy.Adam(3e-4),
            metrics=['accuracy'],
        )

        self.model.summary()
    
    def fit(self, *args):
        return self.model.fit(args)
    
    def evaluate(self, data):
        return self.model.evaluate(data)