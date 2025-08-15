import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

class MSIPatcher(layers.Layer):
    def __init__(self, patch_size, border_size=0, cut_borders=False, channels=14, stride=False,**kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.channels = channels
        self.border_size = border_size

        self.resize = layers.Reshape((-1, patch_size * patch_size, 
                                      self.channels))
        self.cut_borders = cut_borders
        self.stride = stride

    def call(self, images):
        # Create patches from the input images
        if tf.rank(images) == 3:
            images = tf.expand_dims(images, axis=0)
        if not self.stride:
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size - self.border_size*2, self.patch_size - self.border_size*2, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
        else:
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.stride, self.stride, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )

        # Reshape the patches to (batch, num_patches, patch_x, patch_y, patch_channel) and return it.
        patches = self.resize(patches)
        patches = tf.reshape(patches, (patches.shape[0], patches.shape[1], self.patch_size, self.patch_size, self.channels))
        if self.cut_borders:
            inner_patch_size = self.patch_size - self.border_size*2
            patches = tf.cast(tf.squeeze(patches, axis=0), tf.dtypes.float32)

            patches = tf.image.extract_glimpse(patches, size=(inner_patch_size, inner_patch_size), 
                                               offsets=[[-inner_patch_size//2, -inner_patch_size//2]]*patches.shape[0],
                                               centered=True, normalized=False)
            patches = tf.expand_dims(patches, axis=0)
            patches = tf.cast(patches, tf.dtypes.uint8)

        patches = tf.squeeze(patches)    
        return patches

    def __select_image_channel__(self, img, lidx=0):
        channel = img[: , : , lidx]
        channel = tf.expand_dims(channel, axis=-1)
        return channel
        
    def show_patched_image(self, images, layer_idx, patches, vmax=255):
        # This is a utility function which accepts a batch of images and its
        # corresponding patches and help visualize one image and its patches
        # side by side.
        
        idx = np.random.choice(patches.shape[0])
        print(f"Index selected: {idx}.")

        img_channel = self.__select_image_channel__(images[idx], layer_idx)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(keras.utils.array_to_img(img_channel, scale=False), cmap='gray', vmin=0, vmax=vmax)
        plt.axis("off")
        plt.show()

        nx = int((img_channel.shape[0] - self.border_size * 2) / (self.patch_size - self.border_size * 2)) 
        ny =  int((img_channel.shape[1] - self.border_size * 2) / (self.patch_size - self.border_size * 2))
        patches = patches[:,:,:,:,layer_idx]
        plt.figure(figsize=(8, 8))
        for i, patch in enumerate(patches[idx]):
            ax = plt.subplot(nx, ny, i + 1)
            patch_img = patch
            plt.imshow(keras.utils.img_to_array(patch_img),cmap='gray', vmin=0, vmax=vmax)
            plt.axis("off")
        plt.show()

        # Return the index chosen to validate it outside the method.
        return idx

    # taken from https://stackoverflow.com/a/58082878/10319735
    def reconstruct_from_patch(self, patch, img_size, layer_idx):
        # This utility function takes patches from a *single* image and
        # reconstructs it back into the image. This is useful for the train
        # monitor callback.
        print(patch.shape, 'patch shape')
        print(img_size, 'img size')
        num_patches = patch.shape[0]
        nx = int( (img_size[0] - self.border_size * 2) / self.patch_size) 
        ny =  int( (img_size[1] - self.border_size * 2) / self.patch_size)
        #nx = 94
        patch = patch[:,:,:,layer_idx]
        rows = tf.split(patch, nx, axis=0)
        rows = [tf.concat(tf.unstack(x), axis=1) for x in rows]
        reconstructed = tf.concat(rows, axis=0)
        return reconstructed


def msi2patches(msi_data_dict:dict, patch_size:int, border_size:int, remove_unlabeled:bool, device='CPU:0') -> dict:
    with tf.device(device):
        img_patcher = MSIPatcher(patch_size=patch_size, border_size=border_size, cut_borders=False)
        mask_patcher = MSIPatcher(patch_size=patch_size, border_size=border_size, cut_borders=True, channels=1)

        msi_data_dict = {k:{'msi_patches': img_patcher(v['msi']), 'mask':v['mask'], 'path':v['path']}\
                         for k, v in msi_data_dict.items()}
        msi_data_dict = {k:{'msi_patches': v['msi_patches'], 'mask_patches':mask_patcher(v['mask']), 'path':v['path']}\
                         for k, v in msi_data_dict.items()}
        if remove_unlabeled:
            msi_data_dict = {k:remove_unlabeled_patches(v) for k,v in msi_data_dict.items()}
    
    return msi_data_dict

def remove_unlabeled_patches(msi_dict_value):
     #remove patches with no labels
    mask_patches = msi_dict_value['mask_patches']
    msi_patches = msi_dict_value['msi_patches']
    
    indices = tf.where(tf.not_equal(tf.squeeze(mask_patches), tf.constant(0, dtype=tf.uint8)))
    indices, _ = tf.unique(indices[:,0])

    mask_patches = tf.gather(mask_patches, indices=indices)
    mask_patches = tf.squeeze(mask_patches)

    msi_patches = tf.gather(msi_patches, indices=indices)
    
    msi_dict_value['mask_patches'] = mask_patches
    msi_dict_value['msi_patches'] = msi_patches

    return msi_dict_value