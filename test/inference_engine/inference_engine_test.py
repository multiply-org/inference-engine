from multiply_inference_engine import _get_mask_data_set_and_reprojection, infer_kaska_s2
try:
    from osgeo import gdal
except ImportError:
    import gdal
import numpy as np
import os
import pytest
import shutil


def test_get_mask_data_set_and_reprojection_from_state_mask():
    path_to_state_mask = './test/test_data/Barrax_pivots.tif'

    mask_data_set, reprojection = _get_mask_data_set_and_reprojection(path_to_state_mask, None, None)

    assert mask_data_set is not None
    mask = mask_data_set.ReadAsArray().astype(np.bool)
    assert (204, 235) == mask.shape
    assert np.bool == mask.dtype
    assert (576452.584549, 4322656.153353, 578802.584549, 4324696.153353) == reprojection._bounds
    assert 10.0 == pytest.approx(reprojection._x_res)
    assert -10.0 == pytest.approx(reprojection._y_res)


def test_get_mask_data_set_and_reprojection_from_resolution_and_roi_and_bounds_grid_and_destination_grid():
    spatial_resolution = 10
    roi = 'POLYGON((-2.1161949380363905	39.06797142213965, -2.0891490454757693	39.067763300005325, ' \
          '-2.0893840350798922	39.04947274570196, -2.1164229523422327	39.049680732964774, ' \
          '-2.1161949380363905	39.06797142213965))'
    bounds_grid = 'EPSG:4326'
    destination_grid = 'EPSG:32632'

    mask_data_set, reprojection = _get_mask_data_set_and_reprojection(None, spatial_resolution, roi,
                                                                      roi_grid=bounds_grid,
                                                                      destination_grid=destination_grid)

    assert mask_data_set is not None
    mask = mask_data_set.ReadAsArray().astype(np.bool)
    assert (177, 262) == mask.shape
    assert np.bool == mask.dtype
    assert (-2.1164229523422327, 39.04947274570196, -2.0891490454757693, 39.06797142213965) == reprojection._bounds
    assert 10.0 == pytest.approx(reprojection._x_res)
    assert 10.0 == pytest.approx(reprojection._y_res)


def test_get_mask_data_set_and_reprojection_from_resolution_and_roi_and_destination_grid():
    spatial_resolution = 10
    roi = 'POLYGON((576452.584549	4324696.153353, 578802.584549	4324696.153353, ' \
          '578802.584549	4322656.153353, 576452.584549	4322656.153353, ' \
          '576452.584549	4324696.153353))'
    destination_grid = 'EPSG:32632'
    mask_data_set, reprojection = _get_mask_data_set_and_reprojection(None, spatial_resolution, roi,
                                                                      destination_grid=destination_grid)

    assert mask_data_set is not None
    mask = mask_data_set.ReadAsArray().astype(np.bool)
    assert (204, 235) == mask.shape
    assert np.bool == mask.dtype
    assert (576452.584549, 4322656.153353, 578802.584549, 4324696.153353) == reprojection._bounds
    assert 10.0 == pytest.approx(reprojection._x_res)
    assert 10.0 == pytest.approx(reprojection._y_res)


def test_get_mask_data_set_and_reprojection_from_resolution_and_roi_and_roi_grid():
    spatial_resolution = 10
    roi = 'POLYGON((-2.1161949380363905	39.06797142213965, -2.0891490454757693	39.067763300005325, ' \
          '-2.0893840350798922	39.04947274570196, -2.1164229523422327	39.049680732964774, ' \
          '-2.1161949380363905	39.06797142213965))'
    bounds_grid = 'EPSG:4326'
    mask_data_set, reprojection = _get_mask_data_set_and_reprojection(None, spatial_resolution, roi,
                                                                      roi_grid=bounds_grid)

    assert mask_data_set is not None
    mask = mask_data_set.ReadAsArray().astype(np.bool)
    assert (208, 234) == mask.shape
    assert np.bool == mask.dtype
    assert (-2.1164229523422327, 39.04947274570196, -2.0891490454757693, 39.06797142213965) == reprojection._bounds
    assert 10.0 == pytest.approx(reprojection._x_res)
    assert 10.0 == pytest.approx(reprojection._y_res)


def test_get_mask_data_set_and_reprojection_from_resolution_and_roi():
    spatial_resolution = 10
    roi = 'POLYGON((-2.1161949380363905	39.06797142213965, -2.0891490454757693	39.067763300005325, ' \
          '-2.0893840350798922	39.04947274570196, -2.1164229523422327	39.049680732964774, ' \
          '-2.1161949380363905	39.06797142213965))'
    mask_data_set, reprojection = _get_mask_data_set_and_reprojection(None, spatial_resolution, roi)

    assert mask_data_set is not None
    mask = mask_data_set.ReadAsArray().astype(np.bool)
    assert (208, 234) == mask.shape
    assert np.bool == mask.dtype
    assert (-2.1164229523422327, 39.04947274570196, -2.0891490454757693, 39.06797142213965) == reprojection._bounds
    assert 10.0 == pytest.approx(reprojection._x_res)
    assert 10.0 == pytest.approx(reprojection._y_res)


def test_infer_kaska_without_tiling():
    start_time = '2017-06-01'
    end_time = '2017-06-10'
    time_step = 10
    datasets_dir = './test/test_data/s2_test_data/'
    output_dir = './test/test_data/output/'
    roi = 'POLYGON((-2.0291040267713925 39.028237232944704, -2.013311180091705 39.028237232944704, ' \
          '-2.013311180091705 39.0175682127747, -2.0291040267713925 39.0175682127747, ' \
          '-2.0291040267713925 39.028237232944704))'
    try:
        infer_kaska_s2(start_time, end_time, time_step, datasets_dir, forward_models=['s2_prosail_kaska'],
                       output_directory=output_dir, state_mask=None, roi=roi, spatial_resolution=60, roi_grid=None,
                       destination_grid=None, tile_index_x=None, tile_index_y=None, tile_width=None, tile_height=None)
        expected_files = ['s2_cab_A2017-06-01.tif', 's2_cab_A2017-06-10.tif', 's2_cb_A2017-06-01.tif',
                          's2_cb_A2017-06-10.tif', 's2_lai_A2017-06-01.tif', 's2_lai_A2017-06-10.tif']
        for file in expected_files:
            file_path = f'{output_dir}/{file}'
            assert os.path.exists(file_path)
            data = gdal.Open(file_path)
            read_data = data.ReadAsArray()
            assert (20, 23) == read_data.shape
            data = None
    finally:
        shutil.rmtree(output_dir)


def test_infer_kaska_with_tiling():
    start_time = '2017-06-01'
    end_time = '2017-06-10'
    time_step = 10
    datasets_dir = './test/test_data/s2_test_data/'
    output_dir = './test/test_data/output_2/'
    roi = 'POLYGON((-2.0291040267713925 39.028237232944704, -2.013311180091705 39.028237232944704, ' \
          '-2.013311180091705 39.0175682127747, -2.0291040267713925 39.0175682127747, ' \
          '-2.0291040267713925 39.028237232944704))'
    try:
        for x in range(3):
            for y in range(2):
                infer_kaska_s2(start_time, end_time, time_step, datasets_dir, forward_models=['s2_prosail_kaska'],
                               output_directory=output_dir, state_mask=None, roi=roi, spatial_resolution=60,
                               roi_grid=None, destination_grid=None,
                               tile_index_x=x, tile_index_y=y, tile_width=10, tile_height=10)
        expected_files = ['s2_cab_A2017-06-01.tif', 's2_cab_A2017-06-10.tif', 's2_cb_A2017-06-01.tif',
                          's2_cb_A2017-06-10.tif', 's2_lai_A2017-06-01.tif', 's2_lai_A2017-06-10.tif']
        for file in expected_files:
            file_path = f'{output_dir}/{file}'
            assert os.path.exists(file_path)
            data = gdal.Open(file_path)
            read_data = data.ReadAsArray()
            assert (20, 23) == read_data.shape
            data = None
    finally:
        shutil.rmtree(output_dir)
