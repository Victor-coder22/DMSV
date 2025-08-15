import cv2
import numpy as np

from pycocotools import mask as maskUtils
from functools import partial
from .annotations import get_id_folder, get_annimages_ids
from .scenes import get_numpy_path


def get_masks_by_image(annotations):
    ann_images_ids = get_annimages_ids(annotations)
    annotation_masks = {}
    for img_id in ann_images_ids:
        annotation_masks[img_id] = [ann for ann in annotations['annotations'] if ann['image_id'] == img_id]
    return annotation_masks


def get_image_id_recording_env(annotations, recording_folders):
    ann_imgs_id_folder = get_id_folder(annotations)
    np_path_and_id = partial(get_numpy_path, recording_folders=recording_folders)
    ann_images_id_path = list(map(np_path_and_id, ann_imgs_id_folder))
    #ann_images_id_path = filter(not_none, ann_images_id_path)

    image_id_recording_env = {id_folder[0] : id_folder[1].split('/')[4] for id_folder  in ann_images_id_path}
    return image_id_recording_env

def get_label_masks(annotations, recording_folders, cut_borders=False, mergewp=False, kernels=None):
    annotation_masks = get_masks_by_image(annotations)
    image_id_recording_env = get_image_id_recording_env(annotations, recording_folders)
    
    
    img_label_masks = {}
    for img_key in annotation_masks.keys():
        label_mask = []
        for mask in annotation_masks[img_key]:
            if mask['attributes']['uncertain']:
                continue
            rle = maskUtils.frPyObjects(mask['segmentation'], mask['segmentation']['size'][0], mask['segmentation']['size'][1])
            decoded = maskUtils.decode(rle)
            ### all masks are 1
            idx = decoded[:,:]==1
            if mergewp and mask['category_id'] >=3:
                decoded[idx] = mask['category_id'] -1
            else:
                decoded[idx] = mask['category_id']
            if cut_borders:
                #kernel = id_recording_env[mask['image_id']]
                recording_env = image_id_recording_env[mask['image_id']]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernels[recording_env])
                decoded = cv2.erode(decoded, kernel)
                ### Erode image here
            label_mask.append(decoded)
        
        label_mask = np.stack(label_mask, axis=-1)
        #label_mask = tf.constant(label_mask).
        label_mask = np.max(label_mask, axis=-1)
        label_mask = np.expand_dims(label_mask, axis=-1)
        # set background to 255 let class_ids start by 0
        label_mask -= 1
        img_label_masks[img_key] = label_mask
    
    return img_label_masks
