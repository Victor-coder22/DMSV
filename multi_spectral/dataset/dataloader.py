from .scenes import get_msi_data
from .masks import get_label_masks
from .annotations import get_annotations, filter_by_folder, get_id_folder



def load_annotated_msis(annotation_path:str, recording_folders, merge_wood_and_paper, cut_borders, mask_kernels=None):
    annotations = get_annotations(annotation_path)
    filtered_annotations = filter_by_folder(annotations, recording_folders)
    scenes = get_msi_data(filtered_annotations, recording_folders)
    masks = get_label_masks(filtered_annotations, recording_folders=recording_folders,\
                             mergewp=merge_wood_and_paper, cut_borders=cut_borders, kernels=mask_kernels)
    dataset = {key:{'msi': scenes[key]['msi'], 'mask': masks[key], 'path':scenes[key]['path']} \
               for key in masks.keys() & scenes.keys()}

    return dataset