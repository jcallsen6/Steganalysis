import argparse
import os
import pickle

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy

import matplotlib.pyplot as plt

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

    return(parser.parse_args())


def create_gens(path, img_size):
    '''
    Create ImageDataGenerators for train, validate, and test from a given path

    param:
    path - str name of dir
    img_size - int size of one side of image (img_size x img_size)

    return:
    tain_gen, val_gen, test_gen - keras ImageDataGenerators
    '''
    subdirs = []

    for subdir, _, _ in os.walk(path):
        # Remove unwanted subdirectories
        if(subdir != path and subdir.count('/') == 1):
            subdirs.append(subdir)

    gens = []
    
    for subdir in subdirs:
        datagen = ImageDataGenerator()
        gens.append(datagen.flow_from_directory(
            subdir,
            target_size=(img_size, img_size),
            batch_size=1,
            class_mode='binary',
            shuffle=False))
    
    return gens


def eval_model(model, gens):
    '''
    Evaluate model comparing performance against different generators

    param:
    model - Keras neural network
    gens - list of Keras ImageDataGenerator

    return:
    None
    '''
    loss_list = []
    auc_list = []
    gen_names = []
    
    acc = BinaryAccuracy()
    bce = BinaryCrossentropy()

    for gen in gens:
        filename = gen.filenames[0]
        first_index = filename.index('.')
        try:
            second_index = filename.index('.', first_index + 1)
            gen_name = filename[first_index + 1: second_index]
            
            if(gen_name == 'steg'):
                gen_name = 'Basic LSB'
        except Exception as e:
            gen_name = 'None'

        gen_names.append(gen_name)
        
        print(f"Evaluating: {gen_name}")

        predictions = model.predict(gen, verbose=1)
        acc.update_state(gen.labels, predictions)
        loss_list.append((bce(gen.labels, predictions).numpy(), acc.result().numpy())) 
        acc.reset_states()

   
    plt.figure('Model Performance vs LSB Type')

    plt.subplot(1, 2, 1)
    plt.bar(gen_names, [loss[0] for loss in loss_list])
    plt.xlabel('LSB Generator Type')
    plt.ylabel('Binary Crossentropy Loss')

    plt.subplot(1, 2, 2)
    plt.bar(gen_names, [loss[1] for loss in loss_list])
    plt.xlabel('LSB Generator Type')
    plt.ylabel('Accuracy')

    plt.show()


if __name__ == '__main__':
    print('Needed file structure: base_dir: subfolders named after gen type: subfolders for class.')
    print('Must have a folder for 0 and 1 in steganography folders')

    args = get_args()

    with open(args.name + '/params', 'rb') as thresh_file:
        params = pickle.load(thresh_file)

    gens = create_gens(
        args.data, params['img_size'])

    model = load_model(args.name + '/model.h5')

    print(model.summary())

    eval_model(model, gens)
