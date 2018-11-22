from abc import ABCMeta, abstractmethod
from datetime import datetime
from multiply_core.util import reproject_image, block_diag
from multiply_prior_engine import PriorEngine
from typing import List, Union
import gdal
import logging
import numpy as np
import os
import scipy.sparse

__author__ = "Tonio Fincke (Brockmann Consult GmbH)"

LOG = logging.getLogger(__name__ + ".inference_prior")


def _check_state_vector_for_nan(mean_state_vector: np.array, state_grid: np.array, matrix: np.array,
                                len_parameters: int) \
        -> (np.array, np.array, np.array):
    nan_indexes = np.unique(np.floor((np.where(np.isnan(mean_state_vector))[0] / len_parameters))).astype(int)
    return _check_for_nan(mean_state_vector, state_grid, matrix, nan_indexes, len_parameters)


def _check_matrix_for_nan(mean_state_vector: np.array, state_grid: np.array, matrix: np.array,
                          len_parameters: int) -> (np.array, np.array, np.array):
    nan_indexes = np.where(np.isnan(matrix))[0]
    return _check_for_nan(mean_state_vector, state_grid, matrix, nan_indexes, len_parameters)


def _check_for_nan(mean_state_vector: np.array, state_grid: np.array, matrix: np.array, nan_indexes: np.array,
                   len_parameters: int) -> (np.array, np.array, np.array):
    if len(nan_indexes) > 0:
        mean_state_vector_nan_indices = \
            np.array([nan_indexes * len_parameters + i for i in range(len_parameters)]).flatten()
        mean_state_vector = np.delete(mean_state_vector, mean_state_vector_nan_indices)
        state_grid_valid_indexes = np.asarray(np.where(state_grid))
        state_grid[tuple(state_grid_valid_indexes[:, tuple(nan_indexes)])] = False
        matrix = np.delete(matrix, nan_indexes, axis=0)
    return mean_state_vector, state_grid, matrix


class InferencePrior(object):
    """A class to wrap access to priors created by the MULTIPLY prior engine to the inference engine."""

    def __init__(self, prior_engine_config_file: str, global_prior_files: List[str], reference_dataset: gdal.Dataset,
                 use_dummy: bool = False):
        """
        This class encapsulates the access to priors produced by the MULTIPLY Prior Engine by either encapsulating
        the whole Prior Engine or by retrieving a number of global prior files that were the output of the Prior Engine.
        :param prior_engine_config_file: A YAML config file to set up the prior engine
        :param global_prior_files: A list of files that contain information on priors. Files must named according to
        :param reference_dataset: A data set with the spatial extent and resolution to which the prior shall be mapped
        format: 'Priors_<name of parameter>_<name of another parameter>_<day of year>_global.vrt.
        """
        if use_dummy:
            LOG.info('Using dummy for debugging purposes')
            self._inference_prior = DummyInferencePrior()
        elif global_prior_files is not None:
            LOG.info('Using global files to access prior information')
            self._inference_prior = PriorFilesInferencePrior(global_prior_files, reference_dataset)
            if prior_engine_config_file is not None and prior_engine_config_file is not '':
                LOG.info('Passing a config file is not necessary when prior files are present')
        elif prior_engine_config_file is not None and prior_engine_config_file is not '':
            self._inference_prior = PriorEngineInferencePrior(prior_engine_config_file, reference_dataset)
        else:
            raise ValueError('Either config for prior engine or list of vrt files must be given as parameter.')
            # This method sets up the MULTIPLY prior engine using a YAML (e.g.)
            # configuration file. It also sets up a mean dictionary that will be index by
            # parameter name, and then by date (each date storing the mean and
            # standard deviation filenames for the given parameter and date)
            # self.prior_engine = PriorEngine(prior_engine_config_file)
            # self.prior_info_mean = {}
            # self.prior_info_mean = {}
            # self.prior_files = global_prior_files

            # def _get_prior_date(self, parameters, date):
            #     """This method gets the prior information for a given set of parameters and
            #     a particular date. """
            # for param in parameters:
            #     mean, std_dev = self.prior_engine(param, date)
            # mean and std_dev are the full path to the global VRT file
            # self.prior_info_mean[param][date] = mean
            # self.prior_info_std[param][date] = std_dev

            # @abstractmethod
            # def get_all_priors(self, parameters: List[str], time_grid: List[str]):
            #     """A method to go through all dates and query the prior information.
            #     Takes a list of parameters, and a time grid (some iterator, list possibly)
            #     and stores the prior in self.prior_info{mean|std}.
            #     """
            # for timestep in time_grid:
            #     self._get_prior_date(parameters, timestep)

    def process_prior(self, parameters: List[str], time: Union[str, datetime], state_grid: np.array,
                      inv_cov: bool = True) -> (np.array, scipy.sparse.coo_matrix):
        return self._inference_prior.process_prior(parameters, time, state_grid, inv_cov)


class _WrappingInferencePrior(metaclass=ABCMeta):

    @abstractmethod
    def process_prior(self, parameters: List[str], time: Union[str, datetime], state_grid: np.array,
                      inv_cov: bool = True) -> (np.array, scipy.sparse.coo_matrix, np.array):
        """
        This method retrieves the requested parameters for the given time and region. For each parameter, it returns
        a mean state vector and covariance matrix which might be inverse. If this method encounters nan-values in the
        prior data, it will exclude them from the state vector and the matrix and edit the mask.
        :param parameters: List of parameters for which prior information is requested
        :param time: The time for which information is requested.
        :param state_grid: A state grid to indicate the region for which information must be retrieved.
        Must be given in the form of a 2D numpy array.
        :param inv_cov: Whether the covariance matrix shall be inverse.
        :return: The mean state vector, a covariance or inverse covariance matrix (depending on the parameter setting)
        and an updated state grid.
        """
        # """This method reads in the prior parameters for a given `time step`,
        # processes them to match the state grid, and returns a mean state
        # vector and an (inverse) sparse covariance matrix.
        # """


class PriorEngineInferencePrior(_WrappingInferencePrior):

    def __init__(self, prior_engine_config_file: str, reference_dataset: gdal.Dataset):
        """
        This class encapsulates the access to priors produced by the MULTIPLY Prior Engine by either encapsulating
        the whole Prior Engine or by retrieving a number of global prior files that were the output of the Prior Engine.
        :param prior_engine_config_file: A YAML config file to set up the prior engine
        format: 'Priors_<name of parameter>_<name of another parameter>_<day of year>_global.vrt.
        """
        self._prior_engine_config_file = prior_engine_config_file
        self._reference_dataset = reference_dataset

    def process_prior(self, parameters: List[str], time: Union[str, datetime], state_grid: np.array,
                      inv_cov: bool = True) -> (np.array, scipy.sparse.coo_matrix, np.array):
        if type(time) is datetime:
            time = datetime.strftime(time, "%Y-%m-%d")
        prior_engine = PriorEngine(datestr=time, variables=parameters, config=self._prior_engine_config_file)
        priors = prior_engine.get_priors()
        num_pixels = state_grid.sum()
        num_params = len(parameters)
        state_vector_shape = num_pixels * num_params
        matrix_shape = (num_pixels, num_params, num_params)
        mean_state_vector = np.empty(shape=state_vector_shape, dtype=np.float32)
        matrix = np.empty(shape=matrix_shape, dtype=np.float32)
        for i, parameter in enumerate(parameters):
            vrt_dataset = list(priors[parameter].values())[0]
            reprojected_vrt_dataset = reproject_image(vrt_dataset, self._reference_dataset)
            mean_state_vector[i::num_params] = reprojected_vrt_dataset.GetRasterBand(1).ReadAsArray()[state_grid]
            mean_state_vector, state_grid, matrix = \
                _check_state_vector_for_nan(mean_state_vector, state_grid, matrix, num_params)
            matrix[:, i, i] = reprojected_vrt_dataset.GetRasterBand(2).ReadAsArray()[state_grid] ** 2
            mean_state_vector, state_grid, matrix = \
                _check_matrix_for_nan(mean_state_vector, state_grid, matrix, num_params)
        if inv_cov:
            matrix[matrix != 0] = 1. / matrix[matrix != 0]
        return mean_state_vector, block_diag(matrix), matrix


class PriorFilesInferencePrior(_WrappingInferencePrior):

    def __init__(self, global_prior_file_paths: List[str], reference_dataset: gdal.Dataset):
        """
        This class encapsulates the access to priors produced by the MULTIPLY Prior Engine by retrieving a number of
        global prior files that were the output of the Prior Engine.
        :param global_prior_file_paths: A list of absolute paths to files that contain information on priors.
        Files must be named according to format: 'Priors_<name of parameter>_<day of year>_global.vrt.
        """
        self._global_prior_file_paths = global_prior_file_paths
        self._global_prior_file_names = []
        for path in self._global_prior_file_paths:
            file = open(path)
            global_prior_file_name = os.path.basename(file.name)
            self._global_prior_file_names.append(global_prior_file_name)
        self._reference_dataset = reference_dataset

    def process_prior(self, parameters: List[str], time: Union[str, datetime], state_grid: np.array,
                      inv_cov: bool = True) -> (np.array, scipy.sparse.coo_matrix, np.array):
        num_pixels = state_grid.sum()
        num_params = len(parameters)
        state_vector_shape = num_pixels * num_params
        matrix_shape = (num_pixels, num_params, num_params)
        mean_state_vector = np.empty(shape=state_vector_shape, dtype=np.float32)
        matrix = np.zeros(shape=matrix_shape, dtype=np.float32)
        if type(time) is str:
            time = datetime.strptime(time, "%Y-%m-%d")
        day_of_year = time.timetuple().tm_yday
        month = time.month
        for i, parameter in enumerate(parameters):
            if parameter == 'sm':
                requested_prior_file_name = '{}_prior_climatology_{:02d}.vrt'.format(parameter, month)
            else:
                requested_prior_file_name = 'Priors_{}_{:03d}_global.vrt'.format(parameter, day_of_year)
            indices = [j for j, global_prior_file_name in enumerate(self._global_prior_file_names)
                       if global_prior_file_name == requested_prior_file_name]
            if len(indices) == 0:
                raise UserWarning('Could not find prior file {}.'.format(requested_prior_file_name))
            else:
                vrt_dataset = gdal.Open(self._global_prior_file_paths[indices[0]])
                reprojected_vrt_dataset = reproject_image(vrt_dataset, self._reference_dataset)
                mean_state_vector[i::num_params] = reprojected_vrt_dataset.GetRasterBand(1).ReadAsArray()[state_grid]
                mean_state_vector, state_grid, matrix = \
                    _check_state_vector_for_nan(mean_state_vector, state_grid, matrix, num_params)
                matrix[:, i, i] = reprojected_vrt_dataset.GetRasterBand(2).ReadAsArray()[state_grid]**2
                mean_state_vector, state_grid, matrix = \
                    _check_matrix_for_nan(mean_state_vector, state_grid, matrix, num_params)
        if inv_cov:
            matrix[matrix != 0] = 1. / matrix[matrix != 0]
        return mean_state_vector, block_diag(matrix), state_grid


class DummyInferencePrior(_WrappingInferencePrior):
    """
    This class is merely a dummy.
    """

    def process_prior(self, parameters: List[str], time: Union[str, datetime], state_grid: np.array,
                      inv_cov: bool = True) -> (np.array, scipy.sparse.coo_matrix):
        num_pixels = state_grid.sum()
        num_params = len(parameters)
        shape = num_pixels * num_params
        matrix_shape = (num_pixels, num_params, num_params)
        mean_state_vector = np.empty(shape=shape, dtype=np.float32)
        matrix = np.empty(shape=matrix_shape, dtype=np.float32)
        return mean_state_vector, block_diag(matrix)
