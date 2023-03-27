try:
    from osgeo import gdal
except ImportError:
    import gdal
import numpy as np
import pytest

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
    state_vector, inv_cov_matrix = inference_prior.process_prior(parameters, 'i actually do not matter for this test',
                                                                 state_grid, True)
    assert 54 == len(state_vector)
    assert (54, 54) == inv_cov_matrix.shape


def test_process_vrt_prior_mask_out_nan():
    reference_data_set = gdal.Open(REFERENCE_FILE)
    inference_prior = InferencePrior('', [GLOBAL_LAI_VRT_FILE, GLOBAL_SM_VRT_FILE, GLOBAL_CAB_VRT_FILE],
                                     reference_data_set)
    parameters = ['lai', 'cab', 'sm']
    state_grid = np.zeros(shape=[10980, 1830], dtype=np.bool8)
    state_grid[6000:6010, 615:625] = True
    state_vector, matrix, state_grid = inference_prior.process_prior(parameters, '2017-03-01', state_grid, True)
    assert state_vector.shape == (150,)
    assert matrix.shape == (150, 150)
    assert state_grid.shape == (10980, 1830)
    valid_indexes = np.array(np.where(state_grid))
    assert valid_indexes.shape == (2, 50)
    np.testing.assert_array_equal(
        [6000, 6000, 6000, 6000, 6000, 6001, 6001, 6001, 6001, 6001, 6002, 6002, 6002, 6002, 6002,
         6003, 6003, 6003, 6003, 6003, 6004, 6004, 6004, 6004, 6004, 6005, 6005, 6005, 6005, 6005,
         6006, 6006, 6006, 6006, 6006, 6007, 6007, 6007, 6007, 6007, 6008, 6008, 6008, 6008, 6008,
         6009, 6009, 6009, 6009, 6009], valid_indexes[0])
    np.testing.assert_array_equal(
        [615, 616, 617, 618, 619, 615, 616, 617, 618, 619, 615, 616, 617, 618, 619, 615, 616, 617,
         618, 619, 615, 616, 617, 618, 619, 615, 616, 617, 618, 619, 615, 616, 617, 618, 619, 615,
         616, 617, 618, 619, 615, 616, 617, 618, 619, 615, 616, 617, 618, 619], valid_indexes[1])


def test_process_vrt_prior():
    reference_data_set = gdal.Open(REFERENCE_FILE)
    inference_prior = InferencePrior('', [GLOBAL_LAI_VRT_FILE, GLOBAL_SM_VRT_FILE, GLOBAL_CAB_VRT_FILE],
                                     reference_data_set)
    parameters = ['lai', 'cab', 'sm']
    state_grid = np.zeros(shape=[10980, 1830], dtype=np.bool8)
    state_grid[1001:1003, 1701:1703] = True
    state_grid[10001:10004, 1001:1004] = True
    state_vector, matrix, state_grid = inference_prior.process_prior(parameters, '2017-03-01', state_grid, True)

    assert 39 == len(state_vector)
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
    np.testing.assert_array_almost_equal(expected_state_mean_vectors, state_vector)
    assert (39, 39) == matrix.shape
    expected_covariance_vectors = np.array([8.00651474e+01, 10000, 2.04022595e+03,
                                            8.00651474e+01, 10000, 2.04022595e+03,
                                            8.00651474e+01, 10000, 2.04022595e+03,
                                            8.00651474e+01, 10000, 2.04022595e+03,
                                            2.17499832e+02, 10000, 2.20197925e+03,
                                            2.17499832e+02, 10000, 2.20197925e+03,
                                            2.17499832e+02, 10000, 2.20197925e+03,
                                            2.17499832e+02, 10000, 2.20197925e+03,
                                            2.17499832e+02, 10000, 2.20197925e+03,
                                            2.17499832e+02, 10000, 2.20197925e+03,
                                            2.17499832e+02, 10000, 2.20197925e+03,
                                            2.17499832e+02, 10000, 2.20197925e+03,
                                            2.17499832e+02, 10000, 2.20197925e+03]
                                            , dtype=np.float32)
    print(np.diag(matrix.toarray()))
    np.testing.assert_array_almost_equal(expected_covariance_vectors, np.diag(matrix.toarray()))
    assert state_grid.shape == (10980, 1830)
    valid_indexes = np.array(np.where(state_grid))
    np.testing.assert_array_equal(
        [1001, 1001, 1002, 1002, 10001, 10001, 10001, 10002, 10002, 10002, 10003, 10003, 10003], valid_indexes[0])
    np.testing.assert_array_equal(
        [1701, 1702, 1701, 1702, 1001, 1002, 1003, 1001, 1002, 1003, 1001, 1002, 1003], valid_indexes[1])


@pytest.mark.skip(reason='Test does currently not work due to code change in prior engine')
def test_process_prior_engine_prior():
    reference_data_set = gdal.Open(REFERENCE_FILE)
    inference_prior = InferencePrior(PRIOR_CONFIG_FILE, None, reference_data_set)
    parameters = ['lai', 'cab', 'sm']
    state_grid = np.zeros(shape=[10980, 1830], dtype=np.bool8)
    state_grid[1001:1003, 1701:1703] = True
    state_grid[10001:10004, 1001:1004] = True
    state_vector, matrix, state_grid = inference_prior.process_prior(parameters, '2017-03-01', state_grid, True)

    assert 39 == len(state_vector)
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
    np.testing.assert_array_almost_equal(expected_state_mean_vectors, state_vector)
    assert (39, 39) == matrix.shape
    expected_covariance_vectors = np.array([8.00651474e+01, 1.19707766e+05, 2.04022595e+03,
                                            8.00651474e+01, 1.19707766e+05, 2.04022595e+03,
                                            8.00651474e+01, 1.19707766e+05, 2.04022595e+03,
                                            8.00651474e+01, 1.19707766e+05, 2.04022595e+03,
                                            2.17499832e+02, 3.50030406e+05, 2.20197925e+03,
                                            2.17499832e+02, 3.50030406e+05, 2.20197925e+03,
                                            2.17499832e+02, 3.50030406e+05, 2.20197925e+03,
                                            2.17499832e+02, 3.50030406e+05, 2.20197925e+03,
                                            2.17499832e+02, 3.50030406e+05, 2.20197925e+03,
                                            2.17499832e+02, 3.50030406e+05, 2.20197925e+03,
                                            2.17499832e+02, 3.50030406e+05, 2.20197925e+03,
                                            2.17499832e+02, 3.50030406e+05, 2.20197925e+03,
                                            2.17499832e+02, 3.50030406e+05, 2.20197925e+03]
                                           , dtype=np.float32)
    np.testing.assert_array_almost_equal(expected_covariance_vectors, np.diag(matrix.toarray()))
    assert state_grid.shape == (10980, 1830)
    valid_indexes = np.array(np.where(state_grid))
    np.testing.assert_array_equal(
        [1001, 1001, 1002, 1002, 10001, 10001, 10001, 10002, 10002, 10002, 10003, 10003, 10003], valid_indexes[0])
    np.testing.assert_array_equal(
        [1701, 1702, 1701, 1702, 1001, 1002, 1003, 1001, 1002, 1003, 1001, 1002, 1003], valid_indexes[1])

