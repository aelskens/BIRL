"""
Low and High resolution Elastix registration method.

**Installation**

    1. Download compiled executables from https://github.com/SuperElastix/elastix/releases
    2. Try to run both executables locally `elastix --help` and `transformix --help`

        * add path to the lib `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/Applications/elastix/lib`
        * define permanent path or copy libraries `cp /elastix/lib/* /usr/local/lib/`

Example (to change)
-------

    1. Perform sample image registration::

        $HOME/Applications/elastix/bin/elastix \
            -f ./data-images/images/artificial_reference.jpg \
            -m ./data-images/images/artificial_moving-affine.jpg \
            -out ./results/elastix \
            -p ./configs/elastix_affine.txt

    2. Besides using `transformix` for deforming images, you can also use `transformix`
        to evaluate the transformation at some points. This means that the input points are specified
        in the fixed image domain, since the transformation direction is from fixed to moving image.
        Perform image/points warping::

        $HOME/Applications/elastix/bin/transformix \
            -tp ./results/elastix/TransformParameters.0.txt \
            -out ./results/elastix \
            -in ./data-images/images/artificial_moving-affine.jpg \
            -def ./data-images/landmarks/artificial_reference.pts

Usage (to change)
-----
Run the basic ANTs registration with original parameters::

    python bm_experiments/bm_elastix.py \
        -t ./data-images/pairs-imgs-lnds_histol.csv \
        -d ./data-images \
        -o ./results \
        -elastix $HOME/Applications/elastix/bin \
        -cfg ./configs/elastix_affine.txt


.. note:: The origin of VTK coordinate system is in left bottom corner of the image.
 Also the first dimension is horizontal (swapped to matplotlib)

.. note:: For proper confirmation see list of Elastix parameters:
 http://elastix.isi.uu.nl/doxygen/parameter.html

.. note:: If you have any complication with Elastix,
 see https://github.com/SuperElastix/elastix/wiki/FAQ

Klein, Stefan, et al. "Elastix: a toolbox for intensity-based medical image registration."
 IEEE transactions on medical imaging 29.1 (2009): 196-205.
 http://elastix.isi.uu.nl/marius/downloads/2010_j_TMI.pdf

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import glob
import logging
import os
import shutil
import sys
import re
import time

import numpy as np
import pandas as pd
from skimage.color import rgb2gray

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.benchmark import ImRegBenchmark
from birl.utilities.data_io import create_folder, save_image, load_image, load_landmarks, save_landmarks, save_landmarks_pts
from birl.utilities.experiments import exec_commands, iterate_mproc_map
from birl.utilities.dataset import REEXP_FOLDER_SCALE
from bm_experiments import bm_comp_perform
from bm_dataset.rescale_tissue_images import wrap_scale_image


class LowHighResElastix(ImRegBenchmark):
    """ Low and High resolution Elastix registration method

    For the app installation details, see module details.

    EXAMPLE (To change)
    -------
    >>> from birl.utilities.data_io import create_folder, update_path
    >>> path_out = create_folder('temp_results')
    >>> fn_path_conf = lambda n: os.path.join(update_path('configs'), n)
    >>> path_csv = os.path.join(update_path('data-images'), 'pairs-imgs-lnds_mix.csv')
    >>> params = {'path_out': path_out,
    ...           'path_table': path_csv,
    ...           'nb_workers': 1,
    ...           'unique': False,
    ...           'path_elastix': '.',
    ...           'path_config': fn_path_conf('elastix_affine.txt')}
    >>> benchmark = BmElastix(params)
    >>> benchmark.run()  # doctest: +SKIP
    >>> shutil.rmtree(path_out, ignore_errors=True)
    """
    #: required experiment parameters
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + ['path_config']
    #: executable for performing image registration
    EXEC_ELASTIX = 'elastix'
    #: executable for performing image/landmarks transformation
    EXEC_TRANSFX = 'transformix'
    #: default name of warped image (the image extension cam be changed in the config file)
    NAME_IMAGE_WARPED = 'result.*'
    #: default name of warped landmarks
    NAME_LNDS_WARPED = 'outputpoints.txt'
    #: command template for image registration
    COMMAND_REGISTRATION = \
        '%(exec_elastix)s' \
        ' -f %(target)s' \
        ' -m %(source)s' \
        ' -out %(output)s' \
        ' -p %(config)s'
    #: command template for image/landmarks transformation
    COMMAND_TRANSFORMATION = \
        '%(exec_transformix)s' \
        ' -tp %(output)s/TransformParameters.0.txt' \
        ' -out %(output)s' \
        ' -in %(source)s' \
        ' -def %(landmarks)s'

    def _prepare(self):
        """ prepare - copy configurations """
        logging.info('-> copy configuration...')
        self._copy_config_to_expt('path_config')

        def _exec_update(executable):
            is_path = p_elatix and os.path.isdir(p_elatix)
            return os.path.join(p_elatix, executable) if is_path else executable

        p_elatix = self.params.get('path_elastix', '')
        if p_elatix and os.path.isdir(p_elatix):
            logging.info('using local executions from: %s', p_elatix)

        self.exec_elastix = _exec_update(self.EXEC_ELASTIX)
        self.exec_transformix = _exec_update(self.EXEC_TRANSFX)

    def _low_res_preprocessing(self, item):
        """ generate (if not already present) low resolution images X1
        from the inputs and convert them into grayscale

        :param dict item: dictionary with regist. params
        :return dict: the same or updated registration info
        """
        logging.debug('.. generate command before registration experiment')
        # set the paths for this experiment
        path_dir = self._get_path_reg_dir(item)
        path_im_ref, path_im_move, _, _ = self._get_paths(item)

        def __path_img(path_img, pproc):
            img_name, img_ext = os.path.splitext(os.path.basename(path_img))
            return os.path.join(path_dir, img_name + '_' + pproc + img_ext)

        def __save_img(col, path_img_new, img):
            col_temp = col + self.COL_IMAGE_EXT_TEMP
            if isinstance(item.get(col_temp), str):
                path_img = self._absolute_path(item[col_temp], destination='expt')
                os.remove(path_img)
            save_image(path_img_new, img)
            return self._relativize_path(path_img_new, destination='path_exp'), col

        def __convert_gray(path_img_col):
            path_img, col = path_img_col
            path_img_new = __path_img(path_img, 'gray')
            __save_img(col, path_img_new, rgb2gray(load_image(path_img)))
            return self._relativize_path(path_img_new, destination='path_exp'), col

        def _get_1X_lum_image(path_img_scale_col):
            path_img, scale, col = path_img_scale_col
            path_img_low_res = re.sub(REEXP_FOLDER_SCALE, f'scale-{scale}pc', path_img).replace('jpg', 'png')
            
            if self.params.get('compute_x1', None) or not os.path.exists(path_img_low_res):
                wrap_scale_image((path_img, scale), image_ext='.png', overwrite=True)

            return __convert_gray((path_img_low_res, col))

        # Get rescaling percentage
        scale = 100 / item['Full scale magnification'] 

        # Fetch or generate low resolution X1 images and convert them into grayscale
        argv_params = [(path_im_ref, scale, self.COL_IMAGE_REF), (path_im_move, scale, self.COL_IMAGE_MOVE)]
        for path_img, col in iterate_mproc_map(_get_1X_lum_image, argv_params, nb_workers=1, desc=None):
            item[col + self.COL_IMAGE_EXT_TEMP] = path_img

        self.params['preprocessing'] = ['low_res_gray']

        return item

    def _generate_regist_command(self, item):
        """ generate the registration command(s)

        :param dict item: dictionary with registration params
        :return str|list(str): the execution commands
        """
        path_dir = self._get_path_reg_dir(item)
        path_im_ref, path_im_move, _, _ = self._get_paths(item)

        cmd = self.COMMAND_REGISTRATION % {
            'exec_elastix': self.exec_elastix,
            'target': path_im_ref,
            'source': path_im_move,
            'output': path_dir,
            'config': self.params['path_config'],
        }
        return cmd

    def _extract_warped_image_landmarks(self, item):
        """ get registration results - warped registered images and landmarks

        :param dict item: dictionary with registration params
        :return dict: paths to warped images/landmarks
        """
        path_dir = self._get_path_reg_dir(item)
        _, path_img_move, path_lnds_ref, _ = self._get_paths(item)
        path_img_warp, path_lnds_warp = None, None
        path_log = os.path.join(path_dir, self.NAME_LOG_REGISTRATION)

        name_lnds = os.path.basename(path_lnds_ref)
        path_lnds_local = save_landmarks_pts(os.path.join(path_dir, name_lnds), load_landmarks(path_lnds_ref))

        # warping the image and points
        cmd = self.COMMAND_TRANSFORMATION % {
            'exec_transformix': self.exec_transformix,
            'source': path_img_move,
            'output': path_dir,
            'landmarks': path_lnds_local,
        }
        exec_commands(cmd, path_logger=path_log, timeout=self.EXECUTE_TIMEOUT)

        # if there is an output image copy it
        path_im_out = glob.glob(os.path.join(path_dir, self.NAME_IMAGE_WARPED))
        if path_im_out:
            path_im_out = sorted(path_im_out)[0]
            _, ext_img = os.path.splitext(path_im_out)
            name_img, _ = os.path.splitext(os.path.basename(path_img_move))
            path_img_warp = os.path.join(path_dir, name_img + ext_img)
            os.rename(path_im_out, path_img_warp)

        path_lnds_out = os.path.join(path_dir, self.NAME_LNDS_WARPED)
        if os.path.isfile(path_lnds_out):
            path_lnds_warp = os.path.join(path_dir, name_lnds)
            lnds = self.parse_warped_points(path_lnds_out)
            save_landmarks(path_lnds_warp, lnds)

        return {
            self.COL_IMAGE_MOVE_WARP: path_img_warp,
            self.COL_POINTS_REF_WARP: path_lnds_warp,
        }

    def _clear_after_registration(self, item):
        """ clean unnecessarily files after the registration

        :param dict item: dictionary with regist. information
        :return dict: the same or updated regist. info
        """
        logging.debug('.. cleaning after registration experiment, remove `output`')
        path_reg_dir = self._get_path_reg_dir(item)

        for ptn in ('output*', 'result*', '*.txt'):
            for p_file in glob.glob(os.path.join(path_reg_dir, ptn)):
                os.remove(p_file)

        return item

    def _perform_registration(self, df_row):
        """ run single registration experiment with all sub-stages

        :param tuple(int,dict) df_row: row from iterated table
        """
        idx, row = df_row
        logging.debug('-> perform single registration #%d...', idx)
        # create folder for this particular experiment
        row['ID'] = idx
        row[self.COL_REG_DIR] = str(idx)
        path_dir_reg = self._get_path_reg_dir(row)
        # check whether the particular experiment already exists and have result
        if self._ImRegBenchmark__check_exist_regist(idx, path_dir_reg):
            return
        create_folder(path_dir_reg)

        time_start = time.time()
        # do some requested pre-processing if required
        row = self._low_res_preprocessing(row)
        row[self.COL_TIME_PREPROC] = (time.time() - time_start) / 60.   ###########change
        row = self._prepare_img_registration(row)
        # if the pre-processing failed, return back None
        if not row:
            return

        # measure execution time
        time_start = time.time()
        row = self._execute_img_registration(row)
        # if the experiment failed, return back None
        if not row:
            return
        # compute the registration time in minutes
        row[self.COL_TIME] = (time.time() - time_start) / 60.
        # remove some temporary images
        row = self._ImRegBenchmark__remove_pproc_images(row)

        row = self._parse_regist_results(row)
        # if the post-processing failed, return back None
        if not row:
            return
        row = self._clear_after_registration(row)

        if self.params.get('visual', False):
            logging.debug('-> visualise results of experiment: %r', idx)
            self.visualise_registration(
                (idx, row),
                path_dataset=self.params.get('path_dataset'),
                path_experiment=self.params.get('path_exp'),
            )

        return row

    @staticmethod
    def extend_parse(arg_parser):
        """ extent the basic arg parses by some extra required parameters

        :return object:
        """
        # SEE: https://docs.python.org/3/library/argparse.html
        arg_parser.add_argument(
            '-elastix',
            '--path_elastix',
            type=str,
            required=False,
            help='path to folder with elastix executables (if they are not directly callable)'
        )
        arg_parser.add_argument(
            '-cfg', '--path_config', required=True, type=str, help='path to the elastic configuration'
        )
        arg_parser.add_argument(
            '--compute_x1', dest='compute_x1', action='store_true', help='whether the low resolution images (x1) should explicitly be recomputed'
        )
        return arg_parser

    @staticmethod
    def parse_warped_points(path_pts, col_name='OutputPoint'):

        def _parse_lists(cell):
            # get just the string with list
            s_list = cell[cell.index(' = ') + 3:].strip()
            # parse the elements and convert to float
            f_list = list(map(float, s_list[1:-1].strip().split(' ')))
            return f_list

        # load the file as table with custom separator and using the first line
        df = pd.read_csv(path_pts, header=None, sep=';')
        # rename columns according content, it it has following stricture `name = value`
        df.columns = [c[:c.index(' = ')].strip() if '=' in c else n for n, c in zip(df.columns, df.iloc[0])]
        # parse the values for selected column
        vals = df[col_name].apply(_parse_lists).values
        # transform collection of list to matrix
        lnds = np.array(list(vals))
        return lnds


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info(__doc__)
    arg_params, path_expt = LowHighResElastix.main()

    if arg_params.get('run_comp_benchmark', False):
        bm_comp_perform.main(path_expt)