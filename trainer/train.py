import math
import tensorflow as tf
from datetime import datetime
import os

def train_model(
    train_dataset,
    val_dataset, 
    model,
    optimizer,
    loss,
    metrics,
    epochs,
    batch_size, 
    filename, 
    model_name, 
    log_dir='logs'
): 
    
    log_dir = os.path.join(log_dir, model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
    )
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "saved_models/"+ model_name +'.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    
    model.compile(
        optimizer=optimizer, 
        loss=loss, 
        metrics=metrics
    )
    
    model.fit(
        x = train_dataset[0],
        y = train_dataset[1],
        validation_data = (val_dataset[0], val_dataset[1]),
        epochs = epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, tensorboard_callback, model_checkpoint]
    )
    
    
    