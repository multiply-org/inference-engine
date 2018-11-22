from multiply_inference_engine import _get_mask_data_set_and_reprojection
import numpy as np
import pytest


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
