import tensorflow as tf
from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3
import parameter

Incep_dp_rate_01 = 0.1

class Model():
    def __init__(self, num_classes):
        self.PT_model = InceptionV3(
            input_shape=(128,128,3),
            include_top=False,
            weights='imagenet'
        )
        for layer in self.PT_model.layers:
            layer.trainable = False
        
        inp = keras.layers.Input(shape=(parameter.IMAGE_SIZE, parameter.IMAGE_SIZE, 3))
        inp = keras.layers.Resizing(128, 128)(inp)
        x = self.PT_model(inp)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(num_classes*4, activation='relu')(x)
        x = keras.layers.Dropout(Incep_dp_rate_01)(x)
        x = keras.layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = keras.models.Model(inp, x)

        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(3e-4),
            metrics=['accuracy'],
        )

        self.model.summary()
    
    def fit(self, data, epochs=10, verbose=2):
        return self.model.fit(data, epochs=epochs, verbose=verbose)
    
    def evaluate(self, data):
        return self.model.evaluate(data)