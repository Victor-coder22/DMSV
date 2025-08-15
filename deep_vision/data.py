import tensorflow as tf
from functools import reduce

def to_tf_dataset(data_dict, batch_size, prefetch, device='CPU:0') -> tf.data.Dataset:
    with tf.device(device):
        data = list(data_dict.values())
        x0 = data[0]['msi_patches']
        y0 = data[0]['mask_patches']
        del data[0]

        x = reduce(lambda v1, v2: tf.concat([v1, v2['msi_patches']], axis=0), data, x0) 
        y = reduce(lambda v1, v2: tf.concat([v1, v2['mask_patches']], axis=0), data, y0)
        
        print('Create Dataset')

        base_ds = tf.data.Dataset.from_tensor_slices((x, y))\
                        .cache()\
                        .shuffle(buffer_size=len(x), reshuffle_each_iteration=True)\
                        .batch(batch_size,  drop_remainder=True, deterministic=True)
        if prefetch:
            base_ds = base_ds.prefetch(tf.data.AUTOTUNE)
        
        return base_ds