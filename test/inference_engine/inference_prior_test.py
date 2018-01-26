import numpy as np
from multiply_inference_engine import InferencePrior

GLOBAL_VRT_FILE = './test_data/Priors_lai_060_global.vrt'

def test_process_dummy_prior():
    inference_prior = InferencePrior('', [], True)
    parameters = ['cogito', 'ergo', 'sum']
    state_grid = np.zeros(shape=[20, 10], dtype=np.bool)
    state_grid[1:4, 1:4] = True
    state_grid[13:16, 5:8] = True
    processed_priors = inference_prior.process_prior(parameters, 'i actually do not matter for this test',
                                                     state_grid, True)
    assert type(processed_priors) is list
    assert 2, len(processed_priors)
    assert 54, processed_priors[0].shape
    assert [54, 54], processed_priors[1].shape


def test_process_vrt_prior():
    inference_prior = InferencePrior('', [GLOBAL_VRT_FILE])
    parameters = ['lai']
    state_grid = np.zeros(shape=[20, 10], dtype=np.bool)
    state_grid[1:4, 1:4] = True
    state_grid[13:16, 5:8] = True
    processed_priors = inference_prior.process_prior(parameters, '2017-03-01', state_grid, True)

    assert type(processed_priors) is list
    assert 2, len(processed_priors)
