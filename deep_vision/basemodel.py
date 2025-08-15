import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers


class BaseModel(keras.Model):
    
    def __init__(self, class_ids, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.acc_tracker = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.class_ids = class_ids
        
        self.classification_reports = [ClassReport(class_id=id) for id in self.class_ids]
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compute_loss(
                y=y,
                y_pred=y_pred,
            )
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        y, y_pred = self.__clear_dirty_labels__(y, y_pred)
        self.acc_tracker.update_state(y, y_pred)
        y_pred = tf.math.argmax(y_pred, axis=-1, output_type=tf.dtypes.int32)
        y = tf.cast(y, tf.dtypes.int32)
        
        for report in self.classification_reports:
            report.update_state(y, y_pred)
        report_results = {}
        macro_precision =[]
        macro_recall = []
        macro_f1score =[]
        for report, id in zip(self.classification_reports, self.class_ids):
            results = report.result()
            report_results.update(results)
            macro_precision.append(results['precision-'+str(id)])
            macro_recall.append(results['recall-'+str(id)])
            macro_f1score.append(results['f1score-'+str(id)])
                       
        return {self.acc_tracker.name: self.acc_tracker.result(),\
                self.loss_tracker.name: self.loss_tracker.result(),\
                'macro_precision' : sum(macro_precision) / (len(macro_precision)),\
                'macro_recall': sum(macro_recall) / len(macro_recall),\
                'macro_f1score': sum(macro_f1score) / len(macro_f1score),\
                **report_results}
    
    
    def __clear_dirty_labels__(self, y, y_pred):
        clean_idx = tf.where(tf.not_equal(y, 255))
        clean_y = tf.gather_nd(y, clean_idx)
        clean_y_pred = tf.gather_nd(y_pred, clean_idx)
        
        return clean_y, clean_y_pred
    
    
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        loss = self.compute_loss(y=y, y_pred=y_pred)
        self.loss_tracker.update_state(loss)
        # Update the metrics.
        y, y_pred = self.__clear_dirty_labels__(y, y_pred)
        self.acc_tracker.update_state(y, y_pred)
        
        y = tf.cast(y, tf.dtypes.int32)
        y_pred = tf.math.argmax(y_pred, axis=-1, output_type=tf.dtypes.int32)
        for report in self.classification_reports:
            report.update_state(y, y_pred)
        
        report_results = {}
        macro_precision =[]
        macro_recall = []
        macro_f1score =[]
        for report, id in zip(self.classification_reports, self.class_ids):
            results = report.result()
            report_results.update(results)
            macro_precision.append(results['precision-'+str(id)])
            macro_recall.append(results['recall-'+str(id)])
            macro_f1score.append(results['f1score-'+str(id)])
                       
        return {self.acc_tracker.name: self.acc_tracker.result(),\
                self.loss_tracker.name: self.loss_tracker.result(),\
                'macro_precision' : sum(macro_precision) / (len(macro_precision)),\
                'macro_recall': sum(macro_recall) / len(macro_recall),\
                'macro_f1score': sum(macro_f1score) / len(macro_f1score),\
                **report_results}
    



class ClassReport(tf.keras.metrics.Metric):

    def __init__(self, class_id, name='class_report', **kwargs):
        super().__init__(name=name,**kwargs)
        self.class_id = class_id
        self.precision = self.add_weight('precision-'+str(class_id), shape=(), initializer="zeros")
        self.recall = self.add_weight('recall-'+str(class_id), shape=(), initializer="zeros")
        self.f1score = self.add_weight('f1score-'+str(class_id), shape=(), initializer="zeros")
    
        self.truepositive = self.add_weight('truepositive-'+str(class_id), shape=(), initializer="zeros")
        self.falsepositive = self.add_weight('falsepositive-'+str(class_id), shape=(), initializer="zeros")
        self.falsenegative = self.add_weight('falsenegative-'+str(class_id), shape=(), initializer="zeros")
    
    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        idx = tf.where(tf.equal(y_pred, self.class_id))
        idx = tf.squeeze(idx, axis=1)
        preds_p = tf.gather(y_pred, idx)
        y_true_p =  tf.gather(y_true, idx)
        tp = tf.reduce_sum(tf.cast(tf.equal(preds_p, y_true_p),dtype=tf.float32))
        fp = tf.cast(tf.shape(y_true_p)[0], dtype=tf.float32) - tp

        idx = tf.where(tf.equal(y_true, self.class_id))
        preds_r = tf.gather(y_pred, idx)
        y_true_r =  tf.gather(y_true, idx)
        fn =  tf.reduce_sum(tf.cast(tf.not_equal(preds_r, y_true_r), dtype=tf.float32))
        
        self.truepositive.assign_add(tp)
        self.falsepositive.assign_add(fp)
        self.falsenegative.assign_add(fn)

        precision = self.truepositive / (self.truepositive + self.falsepositive + tf.keras.backend.epsilon())
        recall = self.truepositive / (self.truepositive + self.falsenegative + tf.keras.backend.epsilon())
        f1score =  2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        self.precision.assign(precision)
        self.recall.assign(recall)
        self.f1score.assign(f1score)


    def result(self):
        return {self.precision.name[:-2]: self.precision, self.recall.name[:-2]:self.recall, self.f1score.name[:-2]: self.f1score,\
                self.truepositive.name[:-2]:self.truepositive, self.falsepositive.name[:-2]:self.falsepositive, self.falsenegative.name[:-2]:self.falsenegative}
