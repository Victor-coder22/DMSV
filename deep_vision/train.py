from tensorflow import keras

def save_best_model(name, init_threshold, monitor="val_accuracy"):
    
    return keras.callbacks.ModelCheckpoint(
        name,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=init_threshold,
    )


def training(train_ds, val_ds, model_creator, epochs, iteration, verbosity, min_acc, model_name, ds_name, early_stopping, savedir,class_weights=None):
    print("####################TRAINING-"+str(iteration)+"####################")
    smc = save_best_model(name=savedir+'models/'+model_name + "_" + ds_name, init_threshold=min_acc)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=early_stopping)
    new_model = model_creator()
    hist = new_model.fit(train_ds, epochs=epochs,
        validation_data=val_ds, 
                callbacks=[smc, early_stopping_callback], 
                verbose=verbosity,
                class_weight=class_weights
    )
    max_val_acc = max(hist.history['val_accuracy'])
    max_val_f1 = max(hist.history['val_macro_f1score'])
    max_val_precision = max(hist.history['val_macro_precision'])
    max_val_recall = max(hist.history['val_macro_recall'])
    print('Max val acc:', max_val_acc)
    print('Max val f1:', max_val_f1)
    print('Max val precision:', max_val_precision)
    print('Max val recall:', max_val_recall)
    return hist.history