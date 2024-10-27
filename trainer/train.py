import math

def train_model(
    train_dataset,
    val_dataset, 
    model,
    optimizer,
    loss,
    metrics,
    epochs,
    batch_size, 
    filename
): 
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
        batch_size=batch_size
    )
    
    model.save(filename)
    
    
    