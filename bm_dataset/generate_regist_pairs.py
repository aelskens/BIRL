"""
Script for generating registration pairs in two schemas

Sample run::

    python generate_regist_pairs.py \
        -i "../output/synth_dataset/*.jpg" \
        -l "../output/synth_dataset/*.csv" \
        -csv ../output/cover.csv --mode each2all

    python bm_dataset/generate_regist_pairs.py \
        -i "$HOME/Medical-data/dataset_CIMA/lung-lesion_1/scale-100pc/*.png" \
        -l "$HOME/Medical-data/dataset_CIMA/lung-lesion_1/scale-100pc/*.csv" \
        -csv $HOME/Medical-data/dataset_CIMA/dataset_CIMA_100pc.csv --mode each2all

Copyright (C) 2016-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import argparse
import glob
import logging
import os
import sys

import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.benchmark import ImRegBenchmark
from birl.utilities.data_io import image_sizes
from birl.utilities.experiments import parse_arg_params

# list of combination options
OPTIONS_COMBINE = ('first2all', 'each2all')
TISSUES_ACQUISITION_SPECS = {
    "lung-lesion": {
        "full_scale_magnification": 40,
        "pixel_size": 0.174
    }, 
    "lung-lobes": {
        "full_scale_magnification": 10,
        "pixel_size": 1.274
    },
    "mammary-gland": {
        "full_scale_magnification": 10,
        "pixel_size": 2.294
    }, 
    "mice-kidney": {
        "full_scale_magnification": 20,
        "pixel_size": 0.227
    }, 
    "COAD": {
        "full_scale_magnification": 10,
        "pixel_size": 0.468
    }, 
    "gastric": {
        "full_scale_magnification": 40,
        "pixel_size": 0.253
    }, 
    "breast": {
        "full_scale_magnification": 40,
        "pixel_size": 0.253
    }, 
    "kidney": {
        "full_scale_magnification": 40,
        "pixel_size": 0.253
    }
}


def arg_parse_params():
    """ parse the input parameters

    :return dict: parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_pattern_images', type=str, help='path to the input image', required=True)
    parser.add_argument('-l', '--path_pattern_landmarks', type=str, help='path to the input landmarks', required=True)
    parser.add_argument('-csv', '--path_csv', type=str, required=True, help='path to coordinate csv file')
    parser.add_argument('-rot', '--path_initial_rotation', type=str, help='path to a .csv file where each row identify the initial rotation to be applied on the moving image for a specific images pair')
    parser.add_argument(
        '--mode',
        type=str,
        required=False,
        help='type of combination of registration pairs',
        default=OPTIONS_COMBINE[0],
        choices=OPTIONS_COMBINE
    )
    args = parse_arg_params(parser, upper_dirs=['path_csv'])
    return args


def remove_unmatching_items(images, landmarks):
    """
    TO DO
    """
    if len(images) == len(landmarks):
        return
    elif len(images) > len(landmarks):
        to_shorten = images
        reference = landmarks
        ext = ["jpg", "png"]
    else:
        to_shorten = landmarks
        reference = images
        ext = ["csv"]

    to_del = list(range(len(to_shorten)))
    for r in reference:
        name = os.path.basename(r).split('.')[0]
        for e in ext:
            f = to_shorten[0].replace(os.path.basename(to_shorten[0]), f"{name}.{e}")
            if f in to_shorten:
                to_del.remove(to_shorten.index(f))
                break


    for index in sorted(to_del, reverse=True):
        del to_shorten[index]


def generate_pairs(path_pattern_imgs, path_pattern_lnds, path_initial_rotation, mode):
    """ generate the registration pairs as reference and moving images

    :param str path_pattern_imgs: path to the images and image name pattern
    :param str path_pattern_lnds: path to the landmarks and its name pattern
    :param str path_initial_rotation: path to the initial rotations
    :param str mode: one of OPTIONS_COMBINE
    :return: DF
    """
    list_imgs = sorted(glob.glob(path_pattern_imgs))
    list_lnds = sorted(glob.glob(path_pattern_lnds))

    remove_unmatching_items(list_imgs, list_lnds)

    if len(list_imgs) != len(list_lnds):
        raise RuntimeError(
            'the list of loaded images (%i) and landmarks (%i) is different length' % (len(list_imgs), len(list_lnds))
        )
    if len(list_imgs) < 2:
        raise RuntimeError('the minimum is 2 elements')
    logging.info('combining list %i files with "%s"', len(list_imgs), mode)

    if path_initial_rotation:
        df_init_rotation = pd.read_csv(path_initial_rotation)

    pairs = [(0, i) for i in range(1, len(list_imgs))]
    if mode == 'each2all':
        pairs += [(i, j) for i in range(1, len(list_imgs)) for j in range(i + 1, len(list_imgs))]

    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])

    reg_pairs = []
    for i, j in pairs:
        rec = dict(zip(ImRegBenchmark.COVER_COLUMNS, (list_imgs[i], list_imgs[j], list_lnds[i], list_lnds[j])))
        tissue, id = os.path.basename(uppath(list_imgs[i], 2)).split('_')
        rec['Full scale magnification'] = TISSUES_ACQUISITION_SPECS[tissue]['full_scale_magnification']
        # pixel size in microns
        rec['Pixel size'] = TISSUES_ACQUISITION_SPECS[tissue]['pixel_size']
        target_img_size, target_img_diag = image_sizes(rec[ImRegBenchmark.COL_IMAGE_REF])
        source_img_size, source_img_diag = image_sizes(rec[ImRegBenchmark.COL_IMAGE_MOVE])
        rec.update({
            ImRegBenchmark.COL_TARGET_IMAGE_SIZE: target_img_size,
            ImRegBenchmark.COL_SOURCE_IMAGE_SIZE: source_img_size,
            ImRegBenchmark.COL_TARGET_IMAGE_DIAGONAL: target_img_diag,
            ImRegBenchmark.COL_SOURCE_IMAGE_DIAGONAL: source_img_diag
        })
        rec.update({
            ImRegBenchmark.COL_INITIAL_ROTATION: 0
        })

        rotation = 0
        if path_initial_rotation:
            tissue_group = df_init_rotation[df_init_rotation['Tissue'] == f'{tissue}_{id}'].reset_index()
            for r in range(tissue_group.shape[0]):
                if sorted([tissue_group['Target image'][r], tissue_group['Source image'][r]]) == sorted([os.path.basename(im).split('.')[0] for im in [list_imgs[i], list_imgs[j]]]):
                    rotation = tissue_group['Initial rotation'][r]
                    break
        rec.update({
            ImRegBenchmark.COL_INITIAL_ROTATION: rotation
        })

        reg_pairs.append(rec)
    df_overview = pd.DataFrame(reg_pairs)
    return df_overview


def main(path_pattern_images, path_pattern_landmarks, path_csv, path_initial_rotation=None, mode='all2all'):
    """ main entry point

    :param str path_pattern_images: path to images
    :param str path_pattern_landmarks: path to landmarks
    :param str path_csv: path output cover table, add new rows if it exists
    :param str mode: option first2all or all2all
    """
    # if the cover file exist continue in it, otherwise create new
    if os.path.isfile(path_csv):
        logging.info('loading existing csv file: %s', path_csv)
        df_overview = pd.read_csv(path_csv, index_col=0)
    else:
        logging.info('creating new cover file')
        df_overview = pd.DataFrame()

    df_ = generate_pairs(path_pattern_images, path_pattern_landmarks, path_initial_rotation, mode)
    df_overview = pd.concat((df_overview, df_), axis=0)  # , sort=True
    df_overview = df_overview[list(ImRegBenchmark.COVER_COLUMNS_EXT)].reset_index(drop=True)

    logging.info('saving csv file with %i records \n %s', len(df_overview), path_csv)
    df_overview.to_csv(path_csv)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg_params = arg_parse_params()
    logging.info('running...')
    main(**arg_params)
    logging.info('DONE')
