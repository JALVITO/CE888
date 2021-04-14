from config import LEARNING_RATE, SAVED_MODELS_PATH, EPOCHS, DIM, PATIENCE

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Layer, Dense, Dropout, GlobalMaxPool2D, Input
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ----------------------------------------------------
# Keras Custom Layers for Image Augmentation
# ----------------------------------------------------

class Contrast(Layer):
    def __init__(self, contrast_factor=1.5, **kwargs):
        super().__init__(**kwargs)
        self.contrast_factor = contrast_factor

    def call(self, data):
        # Contrast per pixel: (x - mean) * contrast_factor + mean
        return tf.image.adjust_contrast(data, self.contrast_factor)

class Brightness(Layer):
    def __init__(self, brightness_delta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.brightness_delta = brightness_delta

    def call(self, data):
        # Brightness per pixel: x + brightness_delta
        return tf.image.adjust_brightness(data, self.brightness_delta)

class Saturation(Layer):
    def __init__(self, saturation_factor=1.5, **kwargs):
        super().__init__(**kwargs)
        self.saturation_factor = saturation_factor

    def call(self, data):
        # Saturation per pixel: converted from RGB -> HSV and S * saturation_factor
        return tf.image.adjust_saturation(data, self.saturation_factor)

augment = Sequential([
    Saturation(3),
    Brightness(0.1),
    RandomFlip('horizontal'),
    RandomRotation(0.4),
    RandomZoom(0.2)
])

# ----------------------------------------------------
# CNN Model for Fire/NoFire recognition
# ----------------------------------------------------

class FireModel:
    def __init__(self, base, preprocess):
        '''
        Wrapper class for a CNN model that augments incoming data, uses a pre-trained model as a
        a base and is trained to perform binary classification.

        Class arguments:
        base -- one of the available models from tensorflow.keras.applications
        preprocess -- the corresponding preprocess function from the model's module
        '''
        # Instantiate base model
        base_model = base(include_top=False, input_shape=(DIM,DIM,3))
        
        # Build model
        inputs = Input(shape=(DIM, DIM, 3))
        x = augment(inputs)
        x = preprocess(x)
        x = base_model(x)
        x = GlobalMaxPool2D()(x)
        x = Dropout(0.4)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        # Compile model
        self.model = Model(inputs, outputs, name=base.__name__)
        self.model.compile(optimizer=Adam(lr=LEARNING_RATE),
                           loss="binary_crossentropy",
                           metrics=['accuracy'])

    def train(self, train_data, val_data):
        # Stops learning if val_loss begins to decrease
        early_stopping = EarlyStopping(monitor='val_loss',
                                       restore_best_weights=True,
                                       patience=PATIENCE)
        
        # Saves model when obtaining best val_loss
        checkpoint = ModelCheckpoint(SAVED_MODELS_PATH + self.model.name,
                                     save_best_only=True)

        # Train internal model
        return self.model.fit(train_data, 
                              epochs=EPOCHS, 
                              validation_data=val_data,
                              callbacks=[early_stopping, checkpoint])

    def evaluate(self, test_data):
        # Runs evaluation of internal model
        return self.model.evaluate(test_data)

    def summary(self):
        # Returns summary of internal model
        return self.model.summary()