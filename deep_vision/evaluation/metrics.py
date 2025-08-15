import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


def get_gt_and_predicted_labels(dataset, model):
    gt_labels = []
    predicted_labels = []
    for px_batch, label_batch in dataset:
        gt_labels.append(label_batch)
        predictions = model.predict(px_batch, verbose=0)
        #print(predictions.shape)
        pl = tf.argmax(predictions, axis=3)
        #print(pl.shape)
        predicted_labels.append(pl)
    
    with tf.device('CPU:0'):
        gt_labels = tf.concat(gt_labels, axis=0)
        gt_labels = tf.reshape(gt_labels, [-1])
        
        predicted_labels = tf.concat(predicted_labels, axis=0)
        predicted_labels = tf.reshape(predicted_labels, [-1])
    
    return gt_labels, predicted_labels


def get_classification_report(gt_labels, predicted_labels, label_names, as_dict=False):
    # only select labeld regions
    with tf.device('CPU:0'):
        idx = tf.where(tf.not_equal(gt_labels, 255))
        gt_labels = tf.squeeze(tf.gather(gt_labels, idx))
        predicted_labels = tf.squeeze(tf.gather(predicted_labels, idx))
    
    if not as_dict:
        cr = classification_report(gt_labels, predicted_labels, zero_division=0.0,\
                            target_names=[label_names[i] for i in set(gt_labels.numpy())])
        return cr
    else:
        cr = classification_report(gt_labels, predicted_labels, output_dict=True, zero_division=0.0,\
                            target_names=[label_names[i] for i in set(gt_labels.numpy())])
        return cr


def print_and_plot_confusion_matrix(gt_labels, predicted_labels, label_names, suffix, savedir):
    with tf.device('CPU:0'):
        # only select labeld regions
        idx = tf.where(tf.not_equal(gt_labels, 255))
        gt_labels = tf.squeeze(tf.gather(gt_labels, idx))
        predicted_labels = tf.squeeze(tf.gather(predicted_labels, idx))
    
    fig, ax = plt.subplots(figsize=(10,10))
    cm = ConfusionMatrixDisplay.from_predictions(gt_labels, predicted_labels, ax=ax,
        display_labels=label_names)
    plt.savefig(savedir + '/confusion_matrix_'+suffix+'.png')
    
    print('Confusion Matrix')
    print(cm.confusion_matrix)
    print(cm.display_labels)

    cm2json = {'matrix': cm.confusion_matrix.tolist(), 'labels': cm.display_labels}
    with open(savedir+'confusion_matrix_'+suffix+'.json', 'w') as save_file:
        json.dump(cm2json, save_file)


def plot_histories(history, savedir):
        # copied from https://www.tensorflow.org/tutorials/images/transfer_learning
        #for history in histories:
        acc = history['accuracy']
        val_acc = history['val_accuracy']
        macro_acc = history['macro_recall']
        val_macro_acc = history['val_macro_recall']


        loss = history['loss']
        val_loss = history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.plot(macro_acc, label='Macro Accuracy')
        plt.plot(val_macro_acc, label='Validation Macro Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,3.0])
        plt.title('Training and Validation Loss')
        

        max_val = max(val_acc)
        max_val_index = np.argmax(val_acc)
        train_on_max_val = acc[max_val_index]
        max_val_epoch = max_val_index + 1

        summary_str = f'''highest validation accuracy {max_val * 100:.2f}
        with train accuracy {train_on_max_val * 100:.2f}
        on epoch {max_val_epoch}'''
        print(summary_str)

        plt.xlabel('Epoch \n' + summary_str)
        #plt.show()
        plt.savefig(savedir + "/accuracy_plot.png", bbox_inches='tight')
            
            


def plot_recall(history, savedir):
    recalls = [(metric_name, score) for metric_name, score in history.items() if metric_name.startswith('recall')]
    val_recalls = [(metric_name, score) for metric_name, score in history.items() if metric_name.startswith('val_recall')]
    
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)

    for metric_name, score in recalls:
        plt.plot(score, label=metric_name)
    
    plt.plot(history['macro_recall'], label='macro_recall')
    plt.ylabel('Recall')
    plt.xlabel('')
    plt.xticks(ticks=list(range(len(history['macro_recall']))))
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.title('Classwise Training Recall')

    plt.subplot(2, 1, 2)
     
    for metric_name, score in val_recalls:
        plt.plot(score, label=metric_name)
    
    plt.plot(history['val_macro_recall'], label='val_macro_recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.xticks(ticks=list(range(len(history['val_macro_recall']))))
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.title('Classwise Validation Recall')

    val = history['val_macro_recall']
    max_val = max(val)
    max_val_index = np.argmax(val)
    train_on_max_val = history['macro_recall'][max_val_index]
    max_val_epoch = max_val_index + 1

    summary_str = f'''highest validation macro recall {max_val * 100:.2f}
    with train macro recall {train_on_max_val * 100:.2f}
    on epoch {max_val_epoch}'''
    print(summary_str)

    plt.xlabel('Epoch \n' + summary_str)
    #plt.show()
    plt.savefig(savedir + "/recall_plot.png", bbox_inches='tight', pad_inches=0.0)



def plot_precision(history, savedir):
    precisions = [(metric_name, score) for metric_name, score in history.items() if metric_name.startswith('precision')]
    val_precisions = [(metric_name, score) for metric_name, score in history.items() if metric_name.startswith('val_precision')]
    
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)

    for metric_name, score in precisions:
        plt.plot(score, label=metric_name)
    
    plt.plot(history['macro_precision'], label='macro_precision')
    plt.ylabel('Precision')
    plt.xlabel('')
    plt.xticks(ticks=list(range(len(history['macro_precision']))))
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.title('Classwise Training Precision')

    plt.subplot(2, 1, 2)
     
    for metric_name, score in val_precisions:
        plt.plot(score, label=metric_name)
    
    plt.plot(history['val_macro_precision'], label='val_macro_precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.xticks(ticks=list(range(len(history['macro_precision']))))
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.title('Classwise Validation Precision')
    
    val = history['val_macro_precision']
    max_val = max(val)
    max_val_index = np.argmax(val)
    train_on_max_val = history['macro_precision'][max_val_index]
    max_val_epoch = max_val_index + 1

    summary_str = f'''highest validation macro precision {max_val * 100:.2f}
    with train macro precision {train_on_max_val * 100:.2f}
    on epoch {max_val_epoch}'''
    print(summary_str)

    plt.xlabel('Epoch \n' + summary_str)
    #plt.show()
    plt.savefig(savedir + "/precision_plot.png", bbox_inches='tight', pad_inches=0.0)


def plot_f1score(history, savedir):
    f1scores = [(metric_name, score) for metric_name, score in history.items() if metric_name.startswith('f1score')]
    val_f1scores = [(metric_name, score) for metric_name, score in history.items() if metric_name.startswith('val_f1score')]
    
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)

    for metric_name, score in f1scores:
        plt.plot(score, label=metric_name)
    
    plt.plot(history['macro_f1score'], label='macro_f1score')
    plt.ylabel('F1Score')
    plt.xlabel('')
    plt.xticks(ticks=list(range(len(history['macro_f1score']))))
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.title('Classwise Training F1Score')

    plt.subplot(2, 1, 2)
     
    for metric_name, score in val_f1scores:
        plt.plot(score, label=metric_name)
    
    plt.plot(history['val_macro_f1score'], label='val_macro_f1score')
    plt.ylabel('F1Score')
    plt.xticks(ticks=list(range(len(history['val_macro_f1score']))))
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.title('Classwise Validation F1Score')

    val = history['val_macro_f1score']
    max_val = max(val)
    max_val_index = np.argmax(val)
    train_on_max_val = history['macro_f1score'][max_val_index]
    max_val_epoch = max_val_index + 1

    summary_str = f'''highest validation macro f1score {max_val * 100:.2f}
    with train macro f1score {train_on_max_val * 100:.2f}
    on epoch {max_val_epoch}'''
    print(summary_str)

    plt.xlabel('Epoch \n' + summary_str)
    #plt.show()
    plt.savefig(savedir + "/f1score_plot.png", bbox_inches='tight', pad_inches=0.0)