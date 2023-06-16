import tensorflow as tf
import parameter

class Model():
    def __init__(self):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((parameter.IMAGE_SIZE, parameter.IMAGE_SIZE, 3)),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10),
            ], name='my_model'
        )
        

        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(3e-4),
            metrics=['accuracy'],
        )

        self.model.summary()