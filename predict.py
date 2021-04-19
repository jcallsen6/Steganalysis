import argparse
import pickle
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tempfile
import os
import numpy as np
import math


def get_args():
    '''
    Gets args for this program

    param:
    None

    return:
    parsed arguments object
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-image', help='path to PNG image for prediction', type=str, required=True)
    parser.add_argument('-model_name', help='Name of model', type=str, required=True)
    parser.add_argument('-bpp', help='Bits per pixel to check for', type=float, default=1.0)

    return(parser.parse_args())


if __name__ == '__main__':
    args = get_args()
    
    model = load_model(args.model_name + '/model.h5')

    with open(args.model_name + '/params', 'rb') as thresh_file:
        params = pickle.load(thresh_file)

    print(model.summary())

    # TODO functionize
    
    datagen = ImageDataGenerator()
    img_size = int(params['img_size'])
    tmp_dir = None

    img = Image.open(args.image)
    data = np.array(img)
    
    mask = int(''.join(['1'] * math.ceil(args.bpp)), 2)
    binary = data & mask
    lsb_img = Image.fromarray(binary)
    
    tmp_dir = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(tmp_dir.name, '0'))
    lsb_img.save(os.path.join(tmp_dir.name, '0/predict.png'))
    
    datagen = datagen.flow_from_directory(tmp_dir.name, target_size=(img_size, img_size), shuffle=False, batch_size=1)

    predictions = model.predict(datagen, verbose=1)
    thresh_predictions = predictions >= float(params['threshold'])

    for thresh_prediction, prediction, filename in zip(thresh_predictions, predictions, datagen.filenames):
        print(f"{filename}: {prediction[0]}, {thresh_prediction[0]}")

    tmp_dir.cleanup()
