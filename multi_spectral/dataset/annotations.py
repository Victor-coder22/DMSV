import logging
import json

from copy import deepcopy

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def get_label_names(file: str, merge_wood_paper):
    with open(file) as annotation_file:
        annotations = json.load(annotation_file)

    label_names = [cat['name'] for cat in annotations['categories']]
    if merge_wood_paper:
        del label_names[2]
        label_names[1] = 'wood_paper'
    return label_names


def is_in_folders(folders, key):
    def __is_in__(item):
        return rgb_name_to_folder_names(item[key])[:8] in folders
    return __is_in__

def is_in_ids(ids, key):
    def __is_in__(item):
        return item[key] in ids
    return __is_in__

def get_annotations(file: str):
    with open(file) as annotation_file:
        annotations = json.load(annotation_file)
        logging.info(annotations.keys())
        logging.info(annotations['licenses'])
        logging.info(annotations['info'])
        logging.info(annotations['categories'])
        logging.info(annotations['images'][0])
        logging.info(annotations['images'][0].keys())
        logging.info(annotations['annotations'][0])
        logging.info(annotations['annotations'][0]['attributes'])
        return annotations


def filter_by_folder(annotations, folders):
    annotations = deepcopy(annotations)
    folders = [f[:8] for f in folders]
    annotations['images'] = list(filter(is_in_folders(folders, 'file_name'), annotations['images']))
    ids = [img['id'] for img in annotations['images']]
    annotations['annotations'] = list(filter(is_in_ids(ids, 'image_id'), annotations['annotations']))
    return annotations


def get_id_folder(annotations):
    all_images = [(img['id'], img['file_name']) for img in annotations['images']]
    ann_images_ids = sorted(set([ann['image_id'] for ann in annotations['annotations']]))
    ann_images_id_name = [img_id_name for img_id_name in all_images if img_id_name[0] in ann_images_ids]
    ann_images_id_folder = list(map(cropped_image_name_to_msi_path, ann_images_id_name))
    return ann_images_id_folder


def get_annimages_ids(annotations):
    ann_images_ids = sorted(set([ann['image_id'] for ann in annotations['annotations']]))
    return ann_images_ids


def cropped_image_name_to_msi_path(id_name):
    id, name = id_name
    name = name[len('Cropped_Rgb_'): -len('_nofilter_cropped_0_0.png')]
    return (id, name)

def rgb_name_to_folder_names(rgb_name):
    return rgb_name[len('Cropped_Rgb_'): -len('_nofilter_cropped_0_0.png')]