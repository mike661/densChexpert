import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
import os
import pandas as pd
import numpy as np
from custom_loss import WeightedBinaryCrossentropy
from tensorflow.keras.layers import Dense
from dataframe_processing import csv_preprocess





def get_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer




def main():
    # DIRS
    root = './experiment/'
    experiment_number = 0
    for subdirs, dirs, files in os.walk(root):
        if dirs:
            dirs_array = np.array(dirs, dtype='uint8')
            new_experiment_number = int(dirs_array.max())
            experiment_number = new_experiment_number+1
    experiment_path = os.path.join(root, str(experiment_number))
    output_weights_path = os.path.join(experiment_path, "best_weights-{epoch:003d}-{val_auc:.4f}.h5")
    CSV_PATH = './data-split/'
    train_path = os.path.join(CSV_PATH, 'train.csv')
    val_path = os.path.join(CSV_PATH, 'valid.csv')
    batch_size = 16
    train_pd = csv_preprocess(train_path)
    val_pd = csv_preprocess(val_path)
    
    steps = len(train_pd) // batch_size
    
    if not os.path.isdir(experiment_path):
        os.makedirs(experiment_path)
    # TODO checkpoint
    checkpoint = ModelCheckpoint(
        output_weights_path,
        monitor="val_auc",
        save_weights_only=True,
        #save_best_only=True,
        verbose=1,
        mode="max",
        save_freq=steps//10,
    )

    log_csv = CSVLogger(os.path.join(experiment_path, 'logs.csv'), separator=',')

    callbacks = [
        checkpoint,
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
                          verbose=1, mode="min", min_lr=1e-5),
        log_csv,
    ]

    train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        #horizontal_flip=True, 
        preprocessing_function=tf.keras.applications.densenet.preprocess_input)
    val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input)

    label_names = list(train_pd.columns.values)[1:]

    train_generator = train_data_gen.flow_from_dataframe(
        dataframe=train_pd,
        directory="./images",
        x_col="Path",
        y_col=label_names,
        batch_size=batch_size,
        shuffle=True,
        class_mode="raw",
        target_size=(224, 224),
        interpolation='lanczos',
    )

    val_generator = val_data_gen.flow_from_dataframe(
        dataframe=val_pd,
        directory="./images",
        x_col="Path",
        y_col=label_names,
        batch_size=batch_size,
        shuffle=True,
        class_mode="raw",
        target_size=(224, 224),
        interpolation='lanczos',
    )
    

    img_input = tf.keras.Input(shape=(224, 224, 3))

    optimizer = Adam(learning_rate=0.001)

    auc = tf.keras.metrics.AUC(multi_label=True)

    base_model= tf.keras.applications.DenseNet121(
        include_top=False,
        input_tensor=img_input,
        input_shape=(224, 224, 3),
        weights='imagenet',
        pooling="avg")
    
    #last layer of truncated model
    before_avg_pool = get_layer(base_model, 'pool4_relu')

    x = before_avg_pool.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    full_model = Dense(len(label_names), activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs=img_input, outputs=full_model)

    loss_func = WeightedBinaryCrossentropy(label_names, train_pd, reduction='sum')
    model.compile(loss=loss_func,
                  optimizer=optimizer, metrics=[auc, 'accuracy'])

    print(model.summary())

    model.fit(train_generator,
              steps_per_epoch=steps,
              epochs=5,
              validation_data=val_generator,
              validation_steps=(val_generator.n // val_generator.batch_size),
              callbacks = callbacks,
              )
if __name__ == "__main__":
    main()
