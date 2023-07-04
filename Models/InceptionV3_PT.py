import tensorflow as tf
from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3
import parameter


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
        inp2 = keras.layers.Resizing(128, 128)(inp)
        x = self.PT_model(inp2)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(num_classes*4, activation='relu')(x)
        x = keras.layers.Dropout(parameter.LINEAR_DO_RATE)(x)
        x = keras.layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = keras.models.Model(inp, x, name='InceptionV3_PT')

        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.legacy.Adam(parameter.LEARNING_RATE),
            metrics=['accuracy'],
        )

        self.model.summary()
    
    def fit(self, *args, **kwarg):
        return self.model.fit(*args, **kwarg)
    
    def evaluate(self, *args, **kwarg):
        return self.model.evaluate(*args, **kwarg)
    