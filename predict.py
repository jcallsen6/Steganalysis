import argparse
import pickle
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


def get_args():
    '''
    Gets args for this program

    param:
    None

    return:
    parsed arguments object
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('data', default='data/',
                        help='Relative data path folder containing image(s)', type=str)
    parser.add_argument('-name', help='Name of model', type=str)

    return(parser.parse_args())

# TODO work for image directly as well


if __name__ == '__main__':
    args = get_args()

    model = load_model(args.name + '/model.h5')

    with open(args.name + '/params', 'rb') as thresh_file:
        params = pickle.load(thresh_file)

    print(model.summary())

    datagen = ImageDataGenerator()
    img_size = int(params['img_size'])
    datagen = datagen.flow_from_directory(args.data, target_size=(
        img_size, img_size), shuffle=False, batch_size=1)

    predictions = model.predict(datagen, verbose=1)

    thresh_predictions = predictions >= float(params['threshold'])

    for thresh_prediction, prediction, filename in zip(thresh_predictions, predictions, datagen.filenames):
        print(f"{filename}: {prediction[0]}, {thresh_prediction[0]}")

