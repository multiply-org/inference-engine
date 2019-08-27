import gdal
import numpy as np
import os
from datetime import datetime
from scipy import sparse

from multiply_inference_engine.inference_writer import InferenceWriter

OUTPUT_FOLDER = './test/test_data/inference_output'
GEO_TRANSFORM = (576452.584549, 10.0, 0.0, 4324696.153353, 0.0, -10.0)
PROJECTION = "PROJCS[\"WGS 84 / UTM zone 30N\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\"," \
             "6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]]," \
             "PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433," \
             "AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Transverse_Mercator\"]," \
             "PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",-3]," \
             "PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000]," \
             "PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]]," \
             "AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"32630\"]]"


def test_init_with_wrong_format():
    timestamp = datetime(2017, 11, 6, 10, 23, 54)
    try:
        InferenceWriter(['fgt', 'fgzb'], ['dtfvgb', 'fgt', 'zgh', 'fgzb', 'fdtrxw'], OUTPUT_FOLDER, timestamp,
                        GEO_TRANSFORM, PROJECTION, 10, 10, 'fdgbh')
    except ValueError as ve:
        assert ve.args[0] == 'Format fdgbh is not supported.'


def test_init():
    timestamp = datetime(2017, 11, 6, 10, 23, 54)
    inference_writer = InferenceWriter(['fgt', 'fgzb'], ['dtfvgb', 'fgt', 'zgh', 'fgzb', 'fdtrxw'],
                                       OUTPUT_FOLDER, timestamp, GEO_TRANSFORM, PROJECTION, 10, 10)
    assert inference_writer is not None


def test_dump():
    timestamp = datetime(2017, 1, 11, 10, 23, 54)
    inference_writer = InferenceWriter(['fgt', 'fgzb'], ['dtfvgb', 'fgt', 'zgh', 'fgzb', 'fdtrxw'],
                                       OUTPUT_FOLDER, timestamp, GEO_TRANSFORM, PROJECTION, 2, 1)
    x_analysis = [0., 1., 2., 3., 4., 5., 6., 7., 8., 10.]
    p_analysis_inv_data = [[.0, .1, .2, .3, .4, .5, .6, .7, .8, .10],
                           [.11, .12, .13, .14, .15, .16, .17, .18, .20, .21],
                           [.22, .23, .24, .25, .26, .27, .28, .30, .31, .32],
                           [.33, .34, .35, .36, .37, .38, .40, .41, .42, .43],
                           [.44, .45, .46, .47, .48, .50, .51, .52, .53, .54],
                           [.55, .56, .57, .58, .60, .61, .62, .63, .64, .65],
                           [.66, .67, .68, .70, .71, .72, .73, .74, .75, .76],
                           [.77, .78, .80, .81, .82, .83, .84, .85, .86, .87],
                           [.88, .90, .91, .92, .93, .94, .95, .96, .97, .98],
                           [.0, .1, .2, .3, .4, .5, .6, .7, .8, .10]]
    p_analysis_inv = sparse.coo_matrix(p_analysis_inv_data)
    state_mask = np.full(shape=(1, 2), fill_value=True, dtype=np.bool)
    first_file = f'{OUTPUT_FOLDER}/fgt_A2017011.tif'
    first_unc_file = f'{OUTPUT_FOLDER}/fgt_A2017011_unc.tif'
    second_file = f'{OUTPUT_FOLDER}/fgzb_A2017011.tif'
    second_unc_file = f'{OUTPUT_FOLDER}/fgzb_A2017011_unc.tif'
    files = [first_file, first_unc_file, second_file, second_unc_file]
    try:
        inference_writer.dump_data(None, x_analysis, None, p_analysis_inv, state_mask, 5)
        inference_writer.close()
        _assert_file_content(first_file, np.array([[1., 6.]]))
        _assert_file_content(first_unc_file, np.array([[2.8867514, 1.1704115]]))
        _assert_file_content(second_file, np.array([[3., 8.]]))
        _assert_file_content(second_unc_file, np.array([[1.6666666, 1.0153462]]))
    finally:
        for file in files:
            if os.path.exists(file):
                os.remove(file)


def _assert_file_content(file: str, expected: np.array):
    assert os.path.exists(file)
    dataset = gdal.Open(file)
    data = dataset.ReadAsArray()
    np.testing.assert_array_almost_equal(expected, data)


def test_dump_state():
    timestamp = datetime(2017, 1, 11, 10, 23, 54)
    inference_writer = InferenceWriter(['fgt', 'fgzb'], ['dtfvgb', 'fgt', 'zgh', 'fgzb', 'fdtrxw'],
                                       OUTPUT_FOLDER, timestamp, GEO_TRANSFORM, PROJECTION, 2, 1,
                                       state_folder=OUTPUT_FOLDER)
    x_analysis = [0., 1., 2., 3., 4., 5., 6., 7., 8., 10.]
    p_analysis_inv_data = [[.0, .1, .2, .3, .4, .5, .6, .7, .8, .10],
                           [.11, .12, .13, .14, .15, .16, .17, .18, .20, .21],
                           [.22, .23, .24, .25, .26, .27, .28, .30, .31, .32],
                           [.33, .34, .35, .36, .37, .38, .40, .41, .42, .43],
                           [.44, .45, .46, .47, .48, .50, .51, .52, .53, .54],
                           [.55, .56, .57, .58, .60, .61, .62, .63, .64, .65],
                           [.66, .67, .68, .70, .71, .72, .73, .74, .75, .76],
                           [.77, .78, .80, .81, .82, .83, .84, .85, .86, .87],
                           [.88, .90, .91, .92, .93, .94, .95, .96, .97, .98],
                           [.0, .1, .2, .3, .4, .5, .6, .7, .8, .10]]
    p_analysis_inv = sparse.coo_matrix(p_analysis_inv_data)
    state_mask = np.full(shape=(1, 2), fill_value=True, dtype=np.bool)
    x_analysis_file = f'{OUTPUT_FOLDER}/x_analysis_A2017011.npz'
    p_analysis_file = f'{OUTPUT_FOLDER}/p_analysis_A2017011.npz'
    p_analysis_inv_file = f'{OUTPUT_FOLDER}/p_analysis_inv_A2017011.npz'
    state_mask_file = f'{OUTPUT_FOLDER}/state_mask_A2017011.npz'
    files = [x_analysis_file, p_analysis_file, p_analysis_inv_file, state_mask_file]
    try:
        if not os.path.exists(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER)
        inference_writer.dump_state(timestamp, x_analysis, None, p_analysis_inv, state_mask)
        inference_writer.close()
        _assert_dumped_np_content(x_analysis_file, x_analysis)
        assert not os.path.exists(p_analysis_file)
        _assert_dumped_sp_content(p_analysis_inv_file, p_analysis_inv_data)
        _assert_dumped_np_content(state_mask_file, state_mask)
    finally:
        for file in files:
            if os.path.exists(file):
                os.remove(file)


def _assert_dumped_np_content(file: str, expected: np.array):
    assert os.path.exists(file)
    data = np.load(file)['arr_0']
    np.testing.assert_array_almost_equal(expected, data)


def _assert_dumped_sp_content(file: str, expected: np.array):
    assert os.path.exists(file)
    dataset = sparse.load_npz(file)
    np.testing.assert_array_almost_equal(expected, dataset.toarray())
