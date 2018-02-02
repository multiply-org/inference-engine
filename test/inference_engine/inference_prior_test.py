import gdal
import numpy as np
from multiply_inference_engine import InferencePrior

GLOBAL_LAI_VRT_FILE = './test/test_data/Priors/Priors_lai_060_global.vrt'
GLOBAL_CAB_VRT_FILE = './test/test_data/Priors/Priors_cab_060_global.vrt'
GLOBAL_SM_VRT_FILE = './test/test_data/sm_prior_climatology_03.vrt'
REFERENCE_FILE = './test/test_data/Priors/T32UME_20170910T104021_B10.tiff'
PRIOR_CONFIG_FILE = './test/test_data/Priors/prior_engine_test_config.yml'


def test_process_dummy_prior():
    inference_prior = InferencePrior('', [], None, True)
    parameters = ['cogito', 'ergo', 'sum']
    state_grid = np.zeros(shape=[20, 10], dtype=np.bool)
    state_grid[1:4, 1:4] = True
    state_grid[13:16, 5:8] = True
    processed_priors = inference_prior.process_prior(parameters, 'i actually do not matter for this test',
                                                     state_grid, True)
    assert type(processed_priors) is list
    assert 2 == len(processed_priors)
    assert 54 == len(processed_priors[0])
    assert 54 == len(processed_priors[1])


def test_process_vrt_prior():
    reference_data_set = gdal.Open(REFERENCE_FILE)
    inference_prior = InferencePrior('', [GLOBAL_LAI_VRT_FILE, GLOBAL_SM_VRT_FILE, GLOBAL_CAB_VRT_FILE],
                                     reference_data_set)
    parameters = ['lai', 'cab', 'sm']
    state_grid = np.zeros(shape=[10980, 1830], dtype=np.bool8)
    state_grid[1001:1003, 1701:1703] = True
    state_grid[10001:10004, 1001:1004] = True
    processed_priors = inference_prior.process_prior(parameters, '2017-03-01', state_grid, True)

    assert type(processed_priors) is list
    assert 2 == len(processed_priors)
    assert 39 == len(processed_priors[0])
    expected_state_mean_vectors = np.array([0.78757334, 0.99484605, 0.220177,
                                            0.78757334, 0.99484605, 0.220177,
                                            0.78757334, 0.99484605, 0.220177,
                                            0.78757334, 0.99484605, 0.220177,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518]
                                           , dtype=np.float32)
    np.testing.assert_array_almost_equal(expected_state_mean_vectors, processed_priors[0])
    assert 39 == len(processed_priors[1])
    expected_covariance_vectors = np.array([0.1117579, 0.00289027, 0.0221391,
                                            0.1117579, 0.00289027, 0.0221391,
                                            0.1117579, 0.00289027, 0.0221391,
                                            0.1117579, 0.00289027, 0.0221391,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105]
                                           )
    np.testing.assert_array_almost_equal(expected_covariance_vectors, processed_priors[1])

def test_process_prior_engine_prior():
    reference_data_set = gdal.Open(REFERENCE_FILE)
    inference_prior = InferencePrior(PRIOR_CONFIG_FILE, None, reference_data_set)
    parameters = ['lai', 'cab', 'sm']
    state_grid = np.zeros(shape=[10980, 1830], dtype=np.bool8)
    state_grid[1001:1003, 1701:1703] = True
    state_grid[10001:10004, 1001:1004] = True
    processed_priors = inference_prior.process_prior(parameters, '2017-03-01', state_grid, True)

    assert type(processed_priors) is list
    assert 2 == len(processed_priors)
    assert 39 == len(processed_priors[0])
    expected_state_mean_vectors = np.array([0.78757334, 0.99484605, 0.220177,
                                            0.78757334, 0.99484605, 0.220177,
                                            0.78757334, 0.99484605, 0.220177,
                                            0.78757334, 0.99484605, 0.220177,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518,
                                            0.47024861, 0.59706515, 0.264518]
                                           , dtype=np.float32)
    np.testing.assert_array_almost_equal(expected_state_mean_vectors, processed_priors[0])
    assert 39 == len(processed_priors[1])
    expected_covariance_vectors = np.array([0.1117579, 0.00289027, 0.0221391,
                                            0.1117579, 0.00289027, 0.0221391,
                                            0.1117579, 0.00289027, 0.0221391,
                                            0.1117579, 0.00289027, 0.0221391,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105,
                                            0.06780638, 0.00169024, 0.0213105]
                                           )
    np.testing.assert_array_almost_equal(expected_covariance_vectors, processed_priors[1])
