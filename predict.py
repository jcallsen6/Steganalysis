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

    parser.add_argument(
        '-image', help='path to PNG image for prediction', type=str, required=True)
    parser.add_argument('-model_name', help='Name of model',
                        type=str, required=True)
    parser.add_argument(
        '-bpp', help='Bits per pixel to check for', type=float, default=1.0)

    return(parser.parse_args())


def predict(model_name, bpp, image_name):
    '''
    Predict with a given model on a given image

    param:
    model_name - str to model folder
    bpp - float bits per pixel
    image_name - str to image to predict on

    return:
    raw_output - float from 0-1
    output - boolean true/false
    '''
    model = load_model(model_name + '/model.h5')

    with open(model_name + '/params', 'rb') as thresh_file:
        params = pickle.load(thresh_file)

    tmp_dir = None

    img = Image.open(image_name)
    data = np.array(img)

    mask = int(''.join(['1'] * math.ceil(bpp)), 2)
    binary = data & mask
    lsb_img = Image.fromarray(binary)

    tmp_dir = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(tmp_dir.name, '0'))
    lsb_img.save(os.path.join(tmp_dir.name, '0/predict.png'))

    datagen = ImageDataGenerator()
    img_size = params['img_size']
    datagen = datagen.flow_from_directory(tmp_dir.name, target_size=(
        img_size, img_size), shuffle=False, batch_size=1)

    predictions = model.predict(datagen, verbose=1)
    thresh_predictions = predictions >= float(params['threshold'])

    tmp_dir.cleanup()

    return(predictions[0][0], thresh_predictions[0][0])


if __name__ == '__main__':
    args = get_args()

    raw_output, output = predict(args.model_name, args.bpp, args.image)

    print(f"{args.image}: {raw_output}, {output}")
