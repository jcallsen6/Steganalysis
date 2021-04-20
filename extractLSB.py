'''
Extract only certain bits from images
'''
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import math
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
    parser.add_argument('-bpp', default=1, help='Bits per pixel', type=float)
    parser.add_argument(
        '-threads', help='Number of threads to use for parallel processing', type=int)
    return(parser.parse_args())


# source: https://stackoverflow.com/a/37233621
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


def parse_data(path, bpp):
    '''
    param:
    path - str path to data
    bpp - float bits per pixel

    return:
    None
    '''
    filecounter = len([path for path in walk_dir(path)])

    mask = int(''.join(['1'] * math.ceil(bpp)), 2)

    for filepath in tqdm(walk_dir(path), total=filecounter, unit='files'):
        try:
            img = Image.open(filepath)
            data = np.array(img)
            binary = data & mask
            lsb_img = Image.fromarray(binary)
            lsb_img.save(filepath)
        except Exception as e:
            print(e)
            print(filepath)


if __name__ == '__main__':
    args = get_args()

    for i in range(1, args.threads+1):
        p = multiprocessing.Process(target=parse_data, args=(
            args.data + 'dir_' + str(i), args.bpp))
        p.start()
