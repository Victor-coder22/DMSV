import tensorflow as tf
from tensorflow import keras
from keras import layers
from .basemodel import BaseModel
from .losses import no_background_sparse_categorical_crossentropy as nobg_scc


class LightSpectral(tf.Module):
  def __init__(self, squeeze, expand1x1, depth_multiplier, name=None, padding='same'):
    super().__init__(name=name)
    self.padding = padding
    self.squeeze = layers.Conv2D(squeeze, (1,1), activation='relu')
    self.expand1x1 = layers.Conv2D(expand1x1, (1,1), activation='relu')
    self.expand3x3 = layers.DepthwiseConv2D((3,3), depth_multiplier=depth_multiplier,\
                                            activation='relu', padding=self.padding)
    self.concat = layers.Concatenate(axis=-1) 
  def __call__(self, x):
    s = self.squeeze(x)
    e1 = self.expand1x1(s)
    if self.padding == 'valid':
        e1 = layers.Cropping2D(cropping=(1))(e1)
    e3 = self.expand3x3(s) 
    return self.concat([e1,e3])


# interpolation biliear makes determinist execution possible
# cutted second las layer in head
def derterministic_spectrum_unetb8_small_head(patch_size,loss=nobg_scc, optimizer='adam', mwp=False,**kwargs_compile):
    input_shape=(patch_size, patch_size, 14)
    input = keras.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(input)
    x = layers.Conv2D(96, (2,2), strides=(2,2), activation='relu')(x)
    w = LightSpectral(32, 96, 1, name='spectral')(x)
    x = LightSpectral(32, 96, 1, name='spectral')(w)
    skip = layers.Add()([x,w])
    block1 = LightSpectral(32, 192, 2, name='spectral')(skip)#8x8
    
    w = layers.MaxPool2D((2,2), strides=(2,2))(block1)
    x = LightSpectral(32, 192, 2, name='spectral')(w)
    skip = layers.Add()([x,w])
    w = LightSpectral(48, 288, 2, name='spectral')(skip)
    x = LightSpectral(48, 288, 2, name='spectral')(w)
    skip = layers.Add()([x,w])
    block2 = LightSpectral(64, 384, 2, name='spectral')(skip)#4x4
    
    w = layers.MaxPool2D((2,2), strides=(2,2))(block2)
    x = LightSpectral(64, 384, 2, name='spectral')(w)
    skip = layers.Add()([x,w])#2x2
    
    
    up1 = layers.UpSampling2D(2, interpolation='bilinear')(skip)
    up1 = LightSpectral(64, 384, 2, name='spectral')(up1)
    block2  = layers.Cropping2D(0)(block2)
    up1 = layers.Concatenate()([block2, up1])
    up1 = LightSpectral(64, 384, 2, name='spectral', padding='valid')(up1)

    up2 = layers.UpSampling2D(2, interpolation='bilinear')(up1)
    up2 = LightSpectral(32, 192, 2, name='spectral', padding='valid')(up2)
    block1  = layers.Cropping2D(3)(block1)
    up2 = layers.Concatenate()([block1, up2])
    
    up3 = layers.UpSampling2D(2, interpolation='bilinear')(up2)
    up3 = LightSpectral(24, 74, 1, name='spectral', padding='valid')(up3)

    if mwp:
        num_classes = 5
    else:
        num_classes = 6
    
    out = layers.Conv2D(num_classes, 3, activation="softmax", padding='valid')(up3)
    class_ids = list(range(num_classes))
    spectrum_unet = BaseModel(class_ids=class_ids, inputs=input, outputs=out)
    spectrum_unet.compile(optimizer=optimizer, loss=loss, **kwargs_compile)
    return spectrum_unet


# interpolation biliear makes determinist execution possible
def derterministic_spectrum_unetb8(patch_size,loss=nobg_scc, optimizer='adam', mwp=False,**kwargs_compile):
    input_shape=(patch_size, patch_size, 14)
    input = keras.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(input)
    x = layers.Conv2D(96, (2,2), strides=(2,2), activation='relu')(x)
    w = LightSpectral(32, 96, 1, name='spectral')(x)
    x = LightSpectral(32, 96, 1, name='spectral')(w)
    skip = layers.Add()([x,w])
    block1 = LightSpectral(32, 192, 2, name='spectral')(skip)#8x8
    
    w = layers.MaxPool2D((2,2), strides=(2,2))(block1)
    x = LightSpectral(32, 192, 2, name='spectral')(w)
    skip = layers.Add()([x,w])
    w = LightSpectral(48, 288, 2, name='spectral')(skip)
    x = LightSpectral(48, 288, 2, name='spectral')(w)
    skip = layers.Add()([x,w])
    block2 = LightSpectral(64, 384, 2, name='spectral')(skip)#4x4
    
    w = layers.MaxPool2D((2,2), strides=(2,2))(block2)
    x = LightSpectral(64, 384, 2, name='spectral')(w)
    skip = layers.Add()([x,w])#2x2
    
    
    up1 = layers.UpSampling2D(2, interpolation='bilinear')(skip)
    up1 = LightSpectral(64, 384, 2, name='spectral')(up1)
    block2  = layers.Cropping2D(0)(block2)
    up1 = layers.Concatenate()([block2, up1])
    up1 = LightSpectral(64, 384, 2, name='spectral', padding='valid')(up1)

    up2 = layers.UpSampling2D(2, interpolation='bilinear')(up1)
    up2 = LightSpectral(32, 192, 2, name='spectral', padding='valid')(up2)
    block1  = layers.Cropping2D(3)(block1)
    up2 = layers.Concatenate()([block1, up2])
    
    up3 = layers.UpSampling2D(2, interpolation='bilinear')(up2)
    up3 = LightSpectral(24, 74, 1, name='spectral')(up3)
    up3 = LightSpectral(24, 74, 1, name='spectral', padding='valid')(up3)

    if mwp:
        num_classes = 5
    else:
        num_classes = 6
    
    out = layers.Conv2D(num_classes, 3, activation="softmax", padding='valid')(up3)
    class_ids = list(range(num_classes))
    spectrum_unet = BaseModel(class_ids=class_ids, inputs=input, outputs=out)
    spectrum_unet.compile(optimizer=optimizer, loss=loss, **kwargs_compile)
    return spectrum_unet


def nonderterministic_spectrum_unetb8(patch_size,loss=nobg_scc, optimizer='adam', mwp=False,**kwargs_compile):
    input_shape=(patch_size, patch_size, 14)
    input = keras.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(input)
    x = layers.Conv2D(96, (2,2), strides=(2,2), activation='relu')(x)
    w = LightSpectral(32, 96, 1, name='spectral')(x)
    x = LightSpectral(32, 96, 1, name='spectral')(w)
    skip = layers.Add()([x,w])
    block1 = LightSpectral(32, 192, 2, name='spectral')(skip)#8x8
    
    w = layers.MaxPool2D((2,2), strides=(2,2))(block1)
    x = LightSpectral(32, 192, 2, name='spectral')(w)
    skip = layers.Add()([x,w])
    w = LightSpectral(48, 288, 2, name='spectral')(skip)
    x = LightSpectral(48, 288, 2, name='spectral')(w)
    skip = layers.Add()([x,w])
    block2 = LightSpectral(64, 384, 2, name='spectral')(skip)#4x4
    
    w = layers.MaxPool2D((2,2), strides=(2,2))(block2)
    x = LightSpectral(64, 384, 2, name='spectral')(w)
    skip = layers.Add()([x,w])#2x2
    
    
    up1 = layers.UpSampling2D(2)(skip)
    up1 = LightSpectral(64, 384, 2, name='spectral')(up1)
    block2  = layers.Cropping2D(0)(block2)
    up1 = layers.Concatenate()([block2, up1])
    up1 = LightSpectral(64, 384, 2, name='spectral', padding='valid')(up1)

    up2 = layers.UpSampling2D(2)(up1)
    up2 = LightSpectral(32, 192, 2, name='spectral', padding='valid')(up2)
    block1  = layers.Cropping2D(3)(block1)
    up2 = layers.Concatenate()([block1, up2])
    
    up3 = layers.UpSampling2D(2)(up2)
    up3 = LightSpectral(24, 74, 1, name='spectral')(up3)
    up3 = LightSpectral(24, 74, 1, name='spectral', padding='valid')(up3)
    
    if mwp:
        num_classes = 5
    else:
        num_classes = 6
    
    out = layers.Conv2D(num_classes, 3, activation="softmax", padding='valid')(up3)
    class_ids = list(range(num_classes))
    spectrum_unet = BaseModel(class_ids=class_ids, inputs=input, outputs=out)
    spectrum_unet.compile(optimizer=optimizer, loss=loss, **kwargs_compile)
    return spectrum_unet