'''
Adds steganography to given dataset
'''

import argparse
from stegano import lsb, lsbset
from stegano.lsbset import generators
from tqdm import tqdm
from essential_generators import DocumentGenerator
import os
import multiprocessing
from PIL import Image
import sys


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
    parser.add_argument(
        '-threads', help='Number of threads to use for parallel processing', type=int)
    parser.add_argument('-bpp', default=1.0,
                        help='Bits of hidden data per pixel', type=float)
    parser.add_argument('-generators', nargs='+',
                        help='Names of generators to use')

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
            if('.png' in name and 'steg' not in name):
                yield os.path.abspath(os.path.join(subdir, name))


def add_steg(path, bpp):
    '''
    Add steg to each image in directory

    param:
    path - str path to data folder containing .pngs
    bpp - float desired bits per pixel

    return:
    None
    '''
    filecounter = len([path for path in walk_dir(path)])

    gen = DocumentGenerator()

    if(args.generators):
        lsb_generators = [getattr(generators, entry)
                          for entry in args.generators]

    count = 0

    for filepath in tqdm(walk_dir(path), total=filecounter, unit='files'):
        try:
            # TODO use python-magic, if quicker
            im = Image.open(filepath)
            im_size = im.size[0] * im.size[1]
            # average sentence length is 60 bytes -> 480 bits
            message = '\n'.join([gen.sentence()
                                 for i in range(int(im_size * bpp / 480))])
            # remove non-ascii characters as they mess up steganography
            message = message.encode('ascii', 'ignore').decode()

            if(args.generators):
                # TODO fix these if statements
                # + 1 to add normal lsb
                gen_index = int(count % (len(lsb_generators) + 1))

                if(gen_index == len(lsb_generators)):
                    secret = lsb.hide(filepath, message)
                    secret.save(filepath.replace('.png', '.steg.png'))
                else:
                    lsb_gen = lsb_generators[gen_index]
                    secret = lsbset.hide(filepath, message, lsb_gen())
                    secret.save(filepath.replace(
                        '.png', f".{args.generators[gen_index]}.png"))
            else:
                secret = lsb.hide(filepath, message)
                secret.save(filepath.replace('.png', '.steg.png'))
            count += 1
        except Exception as e:
            print(e)
            print(filepath)


if __name__ == '__main__':
    args = get_args()

    for i in range(1, args.threads+1):
        p = multiprocessing.Process(target=add_steg, args=(
            args.data + 'dir_' + str(i), args.bpp))
        p.start()
