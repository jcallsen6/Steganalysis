import argparse
import os
import pickle

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model, save_model
from keras import callbacks
from keras import layers
from keras import optimizers
from keras import applications

import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

# TODO cite bill predictor


def get_args():
    '''
    Gets args for this program

    param:
    None

    return:
    parsed arguments object
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='data/',
                        help='Relative data path to main data folder', type=str)
    parser.add_argument('-name', help='Name of model', type=str)
    parser.add_argument('-epochs', default=10,
                        help='Number runs over the data to train', type=int)
    parser.add_argument('-batch_size', default=32,
                        help='Batch size for training', type=int)
    parser.add_argument('-new_model', action='store_true',
                        help='Create neural network from scratch')
    parser.add_argument('-learning_rate', default=0.1,
                        help='Initial learning rate for training', type=float)
    parser.add_argument('-img_size', default=224,
                        help='Size of image side', type=int)

    return(parser.parse_args())


# TODO remove flip falses
def create_gens(path, batch_size, img_size):
    '''
    Create ImageDataGenerators for train, validate, and test from a given path

    param:
    path - str name of dir
    batch_size - int batch size for training
    img_size - int size of square image
    return:
    tain_gen, val_gen, test_gen - keras ImageDataGenerators
    '''
    train_datagen = ImageDataGenerator(
        horizontal_flip=False,
        vertical_flip=False)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(path, 'train/'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary')

    val_datagen = ImageDataGenerator(
        horizontal_flip=False,
        vertical_flip=False)

    val_gen = val_datagen.flow_from_directory(
        os.path.join(path, 'val/'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary')

    test_datagen = ImageDataGenerator(
        horizontal_flip=False,
        vertical_flip=False)

    test_gen = test_datagen.flow_from_directory(
        os.path.join(path, 'test/'),
        target_size=(img_size, img_size),
        batch_size=1,
        class_mode='binary',
        shuffle=False)

    return(train_gen, val_gen, test_gen)


def build_model(input_shape, learning_rate):
    '''
    Build neural network

    param:
    input shape - tuple of input size
    learning_rate - float initial learning rate

    return:
    keras model
    '''
    base_model = applications.ResNet50(
        weights='imagenet',
        input_shape=input_shape,
        include_top=False)
    base_model.trainable = False

    input = layers.Input(input_shape)
    base = base_model(input, training=False)

    flatten_layer = layers.Flatten()(base)

    output = layers.Dense(1, activation='sigmoid')(flatten_layer)

    model = Model(inputs=input, outputs=output)
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    return model


def eval_model(model, test_gen):
    '''
    Evaluate model on given generator

    param:
    model - Keras neural network
    test_gen - Keras ImageDataGenerator

    return:
    thresh - model ideal threshold value
    '''
    print('Evaluating model')
    loss = model.evaluate(test_gen, verbose=1)

    print('Generating predictions for test data')
    predictions = model.predict(test_gen, verbose=1)
    fpr, tpr, thresholds = metrics.roc_curve(test_gen.labels, predictions)

    auc = metrics.auc(fpr, tpr)

    plt.figure('ROC')
    plt.plot(fpr, tpr, label='AUC = %0.2f' % auc)
    plt.plot([0, 1], label='Random Guessing')
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    plt.legend()
    plt.show()

    # Cite billPredictor
    with np.errstate(divide='ignore', invalid='ignore'):
        performance = np.true_divide(tpr, fpr)
        performance[np.isinf(performance)] = 0
        max_threshold = thresholds[np.nanargmax(performance)]

    return max_threshold


def plot_history(history):
    '''
    Plots model's training accuracy

    param:
    history - keras History object's .history

    return:
    None
    '''
    plt.figure('Training Accuracy')

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train acc', 'val acc'], loc='upper right')

    plt.show()


if __name__ == '__main__':
    args = get_args()

    train_gen, val_gen, test_gen = create_gens(
        args.data, args.batch_size, args.img_size)

    input_shape = train_gen.target_size
    input_shape = (input_shape[0], input_shape[1], 3)

    if(args.new_model):
        model = build_model(input_shape, args.learning_rate)
    else:
        model = load_model(args.name + '/model.h5')

    print(model.summary())

    model_callbacks = [callbacks.ModelCheckpoint(filepath=args.name + '/model.h5'), callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=args.learning_rate/1000), callbacks.EarlyStopping(patience=7)]

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        verbose=1,
        callbacks=model_callbacks)

    if(args.epochs > 1):
        plot_history(history.history)

    thresh = eval_model(model, test_gen)
    print(f"Ideal thresh: {thresh}")

    with open(args.name + '/params', 'wb') as thresh_file:
        pickle.dump(
            {'threshold': thresh, 'img_size': args.img_size}, thresh_file)
