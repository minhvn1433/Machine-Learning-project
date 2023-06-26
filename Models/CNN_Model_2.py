import tensorflow as tf
import parameter

class Model():
    def __init__(self, num_classes):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((parameter.IMAGE_SIZE, parameter.IMAGE_SIZE, 3)),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(num_classes*4, activation='relu'),
                tf.keras.layers.Dropout(parameter.LINEAR_DO_RATE),
                tf.keras.layers.Dense(num_classes, activation='softmax'),
            ], name='CNN_Model'
        )
        

        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.legacy.Adam(parameter.LEARNING_RATE),
            metrics=['recall'],
        )

        self.model.summary()
        
    def fit(self, *args, **kwarg):
        return self.model.fit(*args, **kwarg)
    
    def evaluate(self, *args, **kwarg):
        return self.model.evaluate(*args, **kwarg)
    