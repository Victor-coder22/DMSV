# %%
# add local packages to pypath
import sys
sys.path.insert(0, "/tf/")

import yaml
import json
import os
from multi_spectral.dataset.dataloader import load_annotated_msis

from deep_vision.train import training
from deep_vision import spectrum_unets
from deep_vision import losses
from deep_vision.data import to_tf_dataset
from deep_vision.evaluation.metrics import get_classification_report, get_gt_and_predicted_labels,\
      print_and_plot_confusion_matrix, plot_histories, plot_recall, plot_precision, plot_f1score

from multi_spectral.dataset.split import train_val
from multi_spectral.dataset.patches import msi2patches
from multi_spectral.dataset.annotations import get_label_names
from functools import partial
from tensorflow import keras
import tensorflow as tf
import numpy as np

# load configs
with open("config/environment.yaml", 'r') as env_yaml:
    env_config = yaml.safe_load(env_yaml)
print(env_config)

with open("config/data.yaml", 'r') as data_yaml:
    data_config = yaml.safe_load(data_yaml)
print(data_config)

with open("config/train.yaml", 'r') as train_yaml:
    train_config = yaml.safe_load(train_yaml,)
print(train_config)

with open('config/model.yaml', 'r') as model_yaml:
    model_config = yaml.safe_load(model_yaml)
print(model_config)

# configure reproducibility
if env_config['seed'] is not None:
    keras.utils.set_random_seed(env_config['seed'])
if env_config['deterministic']:
    tf.config.experimental.enable_op_determinism()
tf.config.run_functions_eagerly(True)


#trainings
for i in range(1, env_config['repetitions'] +1):
    report_dir = 'results-'+str(i)+'/' 
    os.makedirs(report_dir, exist_ok=True)
    sys.stdout = open('results-'+str(i)+'/report.txt','wt')
    
    msi_dict = load_annotated_msis(**data_config['recordings_and_annotations'], recording_folders=data_config['train_and_val_folders']) 
    train, val = train_val(msi_dict, **data_config['train_val_split'])

    print('Train MSI Paths')
    print(*[m['path'] for m in train.values()],sep='\n')
    print('Validation MSI Paths')
    print(*[m['path'] for m in val.values()], sep='\n')

    patched_train = msi2patches(train, **data_config['patches'], remove_unlabeled=True)
    patched_val = msi2patches(val, **data_config['patches'], remove_unlabeled=True)
    
    # TODO print random patches to file
    train_ds = to_tf_dataset(patched_train, **data_config['dataset'], prefetch=True)
    val_ds = to_tf_dataset(patched_val, **data_config['dataset'], prefetch=True)

    base_model_fn = getattr(spectrum_unets, model_config['name'])
    base_model_fn

    loss_fn = getattr(losses, model_config['loss'])()
    configured_model_fn = partial(base_model_fn, data_config['patches']['patch_size'], loss=loss_fn,\
                            run_eagerly=model_config['run_eagerly'], mwp=data_config['recordings_and_annotations']['merge_wood_and_paper'])

    #todo refactor to always one iterations
    history = training(train_ds, val_ds, configured_model_fn, **train_config,\
                    model_name=model_config['name'], ds_name=data_config['name'], iteration=i,\
                    savedir=report_dir, class_weights=None)

    with open(report_dir + 'histories.json', 'w') as hist_file:
        json.dump(history, hist_file)

 
    plot_histories(history, report_dir)
    plot_recall(history, report_dir)
    plot_precision(history, report_dir)
    plot_f1score(history, report_dir)
    
    eval_model = configured_model_fn()
    eval_model.load_weights(report_dir+"models/"+model_config['name']+"_"+data_config['name']).expect_partial()

    val_gts, val_preds = get_gt_and_predicted_labels(dataset=val_ds, model=eval_model)

    label_names = get_label_names(data_config['recordings_and_annotations']['annotation_path'],\
                             merge_wood_paper=data_config['recordings_and_annotations']['merge_wood_and_paper'])
    print(label_names)

    
    val_cr = get_classification_report(val_gts, val_preds, label_names=label_names)
    print("########## VALIDATION CLASSIFICATION REPORT ##########")
    print(val_cr)

    print_and_plot_confusion_matrix(val_gts, val_preds, label_names, suffix='val', savedir=report_dir)

    train_gts, train_preds = get_gt_and_predicted_labels(dataset=train_ds, model=eval_model)
    
    train_cr = get_classification_report(train_gts, train_preds, label_names=label_names)
    print("########## TRAINING CLASSIFICATION REPORT ##########")
    print(train_cr)

    print_and_plot_confusion_matrix(train_gts, train_preds, label_names, suffix='train', savedir=report_dir)


    test_msi_dict = load_annotated_msis(**data_config['recordings_and_annotations'], recording_folders=data_config['test_folders']) 
    test_patches = msi2patches(test_msi_dict, **data_config['patches'], remove_unlabeled=True)
    test_ds = to_tf_dataset(test_patches, **data_config['dataset'], prefetch=True)

    test_gts, test_preds = get_gt_and_predicted_labels(dataset=test_ds, model=eval_model)
    test_cr = get_classification_report(test_gts, test_preds, label_names=label_names)
    print("########## TEST CLASSIFICATION REPORT ##########")
    print(test_cr)

    print_and_plot_confusion_matrix(test_gts, test_preds, label_names, suffix='test',savedir=report_dir)
