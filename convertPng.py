'''
Adds steganography to given dataset
'''

import argparse
from PIL import Image
from tqdm import tqdm
import os
import multiprocessing


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
    parser.add_argument('-threads',
                        help='Number of threads to use for parallel processing', type=int)

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
            if('.jpg' in name):
                yield os.path.abspath(os.path.join(subdir, name))


def convert_png(path):
    '''
    converts folder of jpgs to pngs

    param:
    path - str path to data folder containing .jpg files

    return:
    None
    '''
    filecounter = 0
    for filepath in walk_dir(path):
        filecounter += 1

    for filepath in tqdm(walk_dir(path), total=filecounter, unit='files'):
        try:
            img = Image.open(filepath)
            if(img.mode == 'RGB'):
                img.save(filepath.strip('.jpg') + '.png')
        except Exception as e:
            print(e)
            print(filepath)

        os.remove(filepath)


if __name__ == '__main__':
    args = get_args()

    for i in range(args.threads):
        p = multiprocessing.Process(
            target=convert_png, args=(args.data + 'dir_' + str(i),))
        p.start()
