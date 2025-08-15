import numpy as np

from functools import partial

from .annotations import get_id_folder


def get_numpy_path(id_folder, recording_folders):
    id, folder = id_folder
    folder_timestamp = folder[:8]
    recording_folders_timestamps = [f[:8] for f in recording_folders]
    if folder_timestamp in recording_folders_timestamps:
        folder_index = recording_folders_timestamps.index(folder_timestamp)
        recording_folder_name = recording_folders[folder_index]
        base_prefix = '/tf/datasets/msiv6_recordings/'
        path_to_np = base_prefix+recording_folder_name+'/'+folder+'/registered_scene/'+'registered_scene_'+folder+'.npy'
        return id, path_to_np    

def get_msi_data(annotations, recording_folders):
    ann_imgs_id_folder = get_id_folder(annotations)
    np_path_and_id = partial(get_numpy_path, recording_folders=recording_folders)
    ann_images_id_path = map(np_path_and_id, ann_imgs_id_folder)
    #ann_images_id_path = filter(not_none, ann_images_id_path)
    msi_scenes = {id_path[0]: {'msi': np.load(id_path[1]), 'path':id_path[1]} for id_path in ann_images_id_path}
    return msi_scenes
