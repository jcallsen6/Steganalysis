import argparse
import os


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

    return(parser.parse_args())


# source: https://github.com/jcallsen6/billPredictor/blob/master/processJSON.py
def walk_dir(input_path):
    '''
    Walks through subdirectories to grab all necessary data

    param:
    input_path - str path to parent folder containing all data

    return:
    path - str path to file containing data
    '''
    for subdir, dirs, files in os.walk(input_path):
        for name in files:
            if('.png' in name):
                yield os.path.abspath(os.path.join(subdir, name))


def split(path):
    '''
    Splits given dir into 60 20 20 train val test

    param:
    path - str path to parent dir

    return:
    None
    '''
    for root, dirs, files in os.walk(os.path.join(args.data, path)):
        filecount = 0
        counter = 0

    os.mkdir(os.path.join(args.data, 'train/' + path))
    os.mkdir(os.path.join(args.data, 'val/' + path))
    os.mkdir(os.path.join(args.data, 'test/' + path))

    for name in files:
        filecount += 1

    dir = 'train/' + path

    for name in files:
        os.rename(os.path.join(root, name), os.path.join(
            root, name).replace(path, dir))

        if(counter >= filecount * 0.6 and counter <= filecount * 0.8):
            dir = 'val/' + path
        elif(counter >= filecount * 0.8):
            dir = 'test/' + path

        counter += 1


if __name__ == '__main__':
    args = get_args()

    os.mkdir(os.path.join(args.data, 'train/'))
    os.mkdir(os.path.join(args.data, 'val/'))
    os.mkdir(os.path.join(args.data, 'test/'))

    split('0/')
    split('1/')
