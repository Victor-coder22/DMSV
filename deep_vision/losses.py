from tensorflow import keras
from functools import partial

def no_background_sparse_categorical_crossentropy():
    return partial(keras.losses.sparse_categorical_crossentropy, ignore_class=255)