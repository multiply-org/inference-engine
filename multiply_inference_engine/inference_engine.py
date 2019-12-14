import argparse
import gdal
import glob
import json
import logging
import numpy as np
import os
import osr
import shutil
import scipy.sparse as sp

from multiply_inference_engine.inference_prior import InferencePrior
from multiply_inference_engine.inference_writer import InferenceWriter

from collections import namedtuple
from datetime import datetime, timedelta
from kafka import LinearKalman
from kafka.inference import create_prosail_observation_operator
from kafka.inference.narrowbandSAIL_tools import propagate_LAI_narrowbandSAIL as propagator
from kaska import get_inverter, KaSKA
from kaska.kaska_sar import get_sar, read_sar, do_inversion
from multiply_core.models import get_forward_model
from multiply_core.observations import data_validation, GeoTiffWriter, ObservationsFactory
from multiply_core.util import FileRef, FileRefCreation, Reprojection, get_time_from_string, get_aux_data_provider

from scipy.interpolate import interp1d
from scipy.ndimage import label
from shapely.geometry import Polygon
from shapely.wkt import loads
from skimage.filters import sobel
from typing import List, Optional, Tuple, Union

__author__ = "Tonio Fincke (Brockmann Consult GmbH)"


component_progress_logger = logging.getLogger('ComponentProgress')
component_progress_logger.setLevel(logging.INFO)
component_progress_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
component_progress_logging_handler = logging.StreamHandler()
component_progress_logging_handler.setLevel(logging.INFO)
component_progress_logging_handler.setFormatter(component_progress_formatter)
component_progress_logger.addHandler(component_progress_logging_handler)

other_logger = logging.getLogger('InferenceEngine')
other_logger.setLevel(logging.INFO)
other_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
other_logging_handler = logging.StreamHandler()
other_logging_handler.setLevel(logging.INFO)
other_logging_handler.setFormatter(other_formatter)
other_logger.addHandler(other_logging_handler)


def _get_valid_files(datasets_dir: str) -> List[FileRef]:
    file_refs = []
    file_ref_creation = FileRefCreation()
    found_files = glob.glob(datasets_dir + '/**', recursive=True)
    for found_file in found_files:
        found_file = found_file.replace('\\', '/')
        type = data_validation.get_valid_type(found_file)
        if type is not '':
            file_ref = file_ref_creation.get_file_ref(type, found_file)
            if file_ref is not None:
                logging.getLogger('inference_engine').info('retrieve observations from {}'.format(file_ref.url))
                file_refs.append(file_ref)
    return file_refs


def infer(start_time: Union[str, datetime],
          end_time: Union[str, datetime],
          parameter_list: List[str],
          prior_directory: str,
          datasets_dir: str,
          previous_state_dir: str,
          next_state_dir: str,
          output_directory: str,
          emulators_directory: Optional[str] = None,
          forward_models: Optional[List[str]] = None,
          state_mask: Optional[str] = None,
          roi: Optional[Union[str, Polygon]] = None,
          spatial_resolution: Optional[int] = None,
          roi_grid: Optional[str] = None,
          destination_grid: Optional[str] = None,
          with_profiling: bool = False):
    """
    :param start_time: The start time of the inference period
    :param end_time: The end time of the inference period
    :param parameter_list: The list of bio-physical parameters that shall be inferred.
    :param prior_directory: A directory where the global .vrt-files of the priors are located.
    :param datasets_dir: A directory where the input data is located.
    :param previous_state_dir: A directory where the state from the previous inference period has been saved to.
    :param next_state_dir: A directory to which the state can be saved.
    :param forward_models: The names of the forward models
    :param output_directory: The directory to which the output shall be written.
    :param state_mask: A file that defines both the region of interest and the destination grid to which the output
    shall be written. It has a mask to mask out pixels. If roi and spatial resolution are given, the state mask will be
    reprojected to fit these parameters. If not given, roi, spatial_resolution and destination_grid must be set.
    :param roi: The region of interest, either as shapely Polygon or in a WKT representation.
    :param spatial_resolution: The spatial resolution of the destination grid.
    :param roi_grid: A representation of the spatial reference system in which the roi is given, either as EPSG-code
    or as WKT representation. If not given, it is assumed that the roi is given in the
    destination spatial reference system.
    :param destination_grid: A representation of the spatial reference system in which the output shall be given,
    either as EPSG-code or as WKT representation. If not given, it is assumed that the destination grid is the one
    provided by the state mask.
    :param: with_profiling: If true, the inference will output profiling info after its run.
    :return:
    """
    try:
        if with_profiling:
            import cProfile
            cProfile.runctx('_infer(start_time,end_time,inference_type,parameter_list,prior_directory,datasets_dir,'
                            'previous_state_dir,next_state_dir,emulators_directory,output_directory,state_mask,roi,'
                            'spatial_resolution,roi_grid,destination_grid)', globals(), locals(), None)
        else:
            _infer(start_time, end_time, parameter_list, prior_directory, datasets_dir,
                   previous_state_dir, next_state_dir, emulators_directory, forward_models, output_directory,
                   state_mask, roi, spatial_resolution, roi_grid, destination_grid)
    except BaseException as e:
        import sys
        import traceback
        logging.warning(repr(e))
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.stdout.flush()
        raise e


def _infer(start_time: Union[str, datetime],
           end_time: Union[str, datetime],
           parameter_list: List[str],
           prior_directory: str,
           datasets_dir: str,
           previous_state_dir: str,
           next_state_dir: str,
           emulators_directory: Optional[str],
           forward_models: Optional[List[str]],
           output_directory: str,
           state_mask: Optional[str],
           roi: Optional[Union[str, Polygon]],
           spatial_resolution: Optional[int],
           roi_grid: Optional[str],
           destination_grid: Optional[str]):
    # we assume that time is derived for one time step; or, to be more precise, for one time period (with no
    # intermediate time steps). This time step/time period is described by start time and end time.
    if type(start_time) is str:
        start_time = get_time_from_string(start_time)
    if type(end_time) is str:
        end_time = get_time_from_string(end_time)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if forward_models is None and emulators_directory is not None:
        logging.info('Determining forward model name from emulators directory')
        aux_data_provider = get_aux_data_provider()
        model_metadata_file = f'{emulators_directory}/metadata.json'
        if aux_data_provider.assure_element_provided(model_metadata_file):
            with(open(model_metadata_file, 'r')) as model_file:
                model_metadata = json.load(model_file)
                forward_models = [model_metadata['id']]
                logging.info(f"Determined forward model '{forward_models[0]}' from emulators directory")
        else:
            raise FileNotFoundError(f'Could not find {model_metadata_file}')
    mask_data_set, reprojection = _get_mask_data_set_and_reprojection(state_mask, spatial_resolution, roi, roi_grid,
                                                                      destination_grid)
    mask = mask_data_set.ReadAsArray().astype(np.bool)
    geo_transform = mask_data_set.GetGeoTransform()
    projection = mask_data_set.GetProjection()
    complete_parameter_list = []
    for forward_model_name in forward_models:
        logging.info(f'Checking for forward model {forward_model_name}')
        forward_model = get_forward_model(forward_model_name)
        if forward_model is not None:
            logging.info(f'Forward model {forward_model_name} found')
            model_variables = forward_model.variables
            for model_variable in model_variables:
                if model_variable not in complete_parameter_list:
                    complete_parameter_list.append(model_variable)
        else:
            raise ValueError(f'Could not find {forward_model_name}')
    output = InferenceWriter(parameter_list, complete_parameter_list, output_directory, start_time, geo_transform,
                             projection, mask.shape[1], mask.shape[0], state_folder=next_state_dir)
    prior_files = glob.glob(prior_directory + '/*.vrt')
    inference_prior = InferencePrior('', global_prior_files=prior_files, reference_dataset=mask_data_set)

    file_refs = _get_valid_files(datasets_dir)
    observations_factory = ObservationsFactory()
    observations_factory.sort_file_ref_list(file_refs)
    # an observations wrapper to be passed to kafka
    observations = observations_factory.create_observations(file_refs, reprojection, forward_models)

    p_forecast_inv = None
    x_forecast = None
    if previous_state_dir is not None and os.path.exists(previous_state_dir):
        p_inv_fname = "P_analysis_inv_%s.npz" % start_time.strftime("A%Y%j")
        p_inv_fname = os.path.join(previous_state_dir, p_inv_fname)
        if os.path.exists(p_inv_fname):
            p_forecast_inv = sp.load_npz(p_inv_fname)
        x_fname = "X_analysis_%s.npz" % start_time.strftime("A%Y%j")
        x_fname = os.path.join(previous_state_dir, x_fname)
        if os.path.exists(x_fname):
            x_forecast = np.load(x_fname)['arr_0']
        mask_fname = "state_mask_%s.npz" % start_time.strftime("A%Y%j")
        mask_fname = os.path.join(previous_state_dir, mask_fname)
        if os.path.exists(mask_fname):
            mask = np.load(mask_fname)['arr_0']
    if p_forecast_inv is None or x_forecast is None:
        processed_prior = inference_prior.process_prior(complete_parameter_list, start_time, mask)
        if x_forecast is None:
            x_forecast = processed_prior[0]
        if p_forecast_inv is None:
            p_forecast_inv = processed_prior[1]
        mask = processed_prior[2]

    linear_kalman = LinearKalman(observations, output, mask, create_prosail_observation_operator,
                                 complete_parameter_list, state_propagation=propagator, prior=None, linear=False)

    # Inflation amount for propagation
    q = np.zeros_like(x_forecast)
    # todo figure out correct setting
    if 'lai' in complete_parameter_list:
        lai_index = complete_parameter_list.index('lai')
        q[lai_index::len(complete_parameter_list)] = 0.05
    linear_kalman.set_trajectory_model()
    linear_kalman.set_trajectory_uncertainty(q)

    time_grid = [start_time, end_time]
    linear_kalman.run(time_grid, x_forecast, None, p_forecast_inv, iter_obs_op=True)


def infer_kaska_s2(start_time: Union[str, datetime],
                   end_time: Union[str, datetime],
                   time_step: Union[int, timedelta],
                   datasets_dir: str,
                   forward_models: List[str],
                   output_directory: str,
                   parameters: Optional[List[str]] = None,
                   state_mask: Optional[str] = None,
                   roi: Optional[Union[str, Polygon]] = None,
                   spatial_resolution: Optional[int] = None,
                   roi_grid: Optional[str] = None,
                   destination_grid: Optional[str] = None,
                   tile_index_x: Optional[int] = 0,
                   tile_index_y: Optional[int] = 0,
                   tile_width: Optional[int] = None,
                   tile_height: Optional[int] = None
                   ):
    if type(start_time) is str:
        start_time = get_time_from_string(start_time)
    if type(end_time) is str:
        end_time = get_time_from_string(end_time)
    if type(time_step) is int:
        time_step = timedelta(days=time_step)
    time_grid = []
    current_time = start_time
    while current_time < end_time:
        time_grid.append(current_time)
        current_time += time_step
    time_grid.append(end_time)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    temp_dir = f'{output_directory}/temp/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    mask_data_set, untiled_reprojection = _get_mask_data_set_and_reprojection(state_mask, spatial_resolution, roi,
                                                                              roi_grid, destination_grid)
    reprojection = untiled_reprojection
    tile_mask_data_set = mask_data_set
    raster_width = mask_data_set.RasterXSize
    raster_height = mask_data_set.RasterYSize
    offset_x = 0
    offset_y = 0
    if tile_width is not None and tile_height is not None:
        geo_transform = mask_data_set.GetGeoTransform()
        ulx, xres, xskew, uly, yskew, yres = geo_transform
        minlrx = ulx + (mask_data_set.RasterXSize * xres)
        minlry = uly + (mask_data_set.RasterYSize * yres)
        ulx = ulx + (tile_index_x * tile_width * xres)
        uly = uly + (tile_index_y * tile_height * yres)
        lrx = ulx + (tile_width * xres)
        lry = uly + (tile_height * yres)
        raster_width = tile_width
        raster_height = tile_height
        if (lrx > ulx and lrx > minlrx) or (lrx < ulx and lrx < minlrx):
            lrx = minlrx
            raster_width = np.abs((ulx - lrx) / xres)
        if (lry > uly and lry > minlry) or (lry < uly and lry < minlry):
            lry = minlry
            raster_height = np.abs((uly - lry) / yres)
        offset_x = tile_index_x * tile_width
        offset_y = tile_index_y * tile_height
        roi_bounds = (min(ulx, lrx), min(uly, lry), max(ulx, lrx), max(uly, lry))
        destination_spatial_reference_system = osr.SpatialReference()
        projection = mask_data_set.GetProjection()
        destination_spatial_reference_system.ImportFromWkt(projection)
        reprojection = Reprojection(roi_bounds, xres, yres, destination_spatial_reference_system)
        tile_mask_data_set = reprojection.reproject(mask_data_set)
    elif tile_width is not None or tile_height is not None:
        logging.warning('To use tiling, parameters tileWidth and tileHeight must be set. Continue without tiling')
    file_refs = _get_valid_files(datasets_dir)
    observations_factory = ObservationsFactory()
    observations_factory.sort_file_ref_list(file_refs)
    # an observations wrapper to be passed to kafka
    observations = observations_factory.create_observations(file_refs, reprojection, forward_models)
    model_parameter_names = []
    other_logger.info('Assembling model parameter names')
    for forward_model_name in forward_models:
        forward_model = get_forward_model(forward_model_name)
        if forward_model is None:
            other_logger.warning(f'Could not find forward model {forward_model_name}')
            continue
        for variable in forward_model.variables:
            other_logger.info(f'Checking variable {variable}')
            if variable not in model_parameter_names:
                model_parameter_names.append(variable)
    outfile_names = []
    requested_indexes = []
    for i, parameter_name in enumerate(model_parameter_names):
        other_logger.info(f'Checking for {parameter_name}')
        if parameters is None or parameter_name in parameters:
            other_logger.info(f'Creating output files for {parameter_name}')
            requested_indexes.append(i)
            for time_step in time_grid:
                time = time_step.strftime('%Y-%m-%d')
                outfile_names.append(f"{output_directory}/s2_{parameter_name}_A{time}.tif")
                other_logger.info(f'Created output file {parameter_name}')
    writer = GeoTiffWriter(outfile_names, mask_data_set.GetGeoTransform(), mask_data_set.GetProjection(),
                           mask_data_set.RasterXSize, mask_data_set.RasterYSize, num_bands=None, data_types=None)
    data = []
    # for j, sub_data in enumerate(results[1:]):
    for j in requested_indexes:
        for i in range(len(time_grid)):
            data.append(np.zeros((raster_height, raster_width)))
    if len(observations.dates == 0):
        logging.info('No valid observations found. Will skip inference.')
    else:
        # todo make this more elaborate when more than one inverter is available
        approx_inverter = get_inverter("prosail_5paras", "Sentinel2")

        kaska = KaSKA(observations=observations,
                      time_grid=time_grid,
                      state_mask=tile_mask_data_set,
                      approx_inverter=approx_inverter,
                      output_folder=temp_dir,
                    save_sgl_inversion=False)
        results = kaska.run_retrieval()
        for j, sub_data in enumerate(results[1:]):
            if j in requested_indexes:
                for i in range(len(time_grid)):
                    data.append(sub_data[i, :, :])
    writer.write(data, raster_width, raster_height, offset_x, offset_y)
    writer.close()
    shutil.rmtree(temp_dir)


def infer_kaska_s1(s1_stack_file_dir: str,
                   priors_dir: str,
                   output_directory: str,
                   parameters: Optional[List[str]] = None,
                   state_mask: Optional[str] = None,
                   roi: Optional[Union[str, Polygon]] = None,
                   spatial_resolution: Optional[int] = None,
                   roi_grid: Optional[str] = None,
                   destination_grid: Optional[str] = None,
                   tile_index_x: Optional[int] = 0,
                   tile_index_y: Optional[int] = 0,
                   tile_width: Optional[int] = None,
                   tile_height: Optional[int] = None
                   ):
    mask_data_set, untiled_reprojection = _get_mask_data_set_and_reprojection(state_mask, spatial_resolution, roi,
                                                                              roi_grid, destination_grid)
    reprojection = untiled_reprojection
    tile_mask_data_set = mask_data_set
    raster_width = mask_data_set.RasterXSize
    raster_height = mask_data_set.RasterYSize
    offset_x = 0
    offset_y = 0
    if tile_width is not None and tile_height is not None:
        geo_transform = mask_data_set.GetGeoTransform()
        ulx, xres, xskew, uly, yskew, yres = geo_transform
        minlrx = ulx + (mask_data_set.RasterXSize * xres)
        minlry = uly + (mask_data_set.RasterYSize * yres)
        ulx = ulx + (tile_index_x * tile_width * xres)
        uly = uly + (tile_index_y * tile_height * yres)
        lrx = ulx + (tile_width * xres)
        lry = uly + (tile_height * yres)
        raster_width = tile_width
        raster_height = tile_height
        if (lrx > ulx and lrx > minlrx) or (lrx < ulx and lrx < minlrx):
            lrx = minlrx
            raster_width = np.abs((ulx - lrx) / xres)
        if (lry > uly and lry > minlry) or (lry < uly and lry < minlry):
            lry = minlry
            raster_height = np.abs((uly - lry) / yres)
        offset_x = tile_index_x * tile_width
        offset_y = tile_index_y * tile_height
        roi_bounds = (min(ulx, lrx), min(uly, lry), max(ulx, lrx), max(uly, lry))
        destination_spatial_reference_system = osr.SpatialReference()
        projection = mask_data_set.GetProjection()
        destination_spatial_reference_system.ImportFromWkt(projection)
        reprojection = Reprojection(roi_bounds, xres, yres, destination_spatial_reference_system)
        tile_mask_data_set = reprojection.reproject(mask_data_set)
    elif tile_width is not None or tile_height is not None:
        logging.warning('To use tiling, parameters tileWidth and tileHeight must be set. Continue without tiling')

    s1_stack_file = glob.glob(os.path.join(s1_stack_file_dir, '*.nc'))[0]
    component_progress_logger.info('14')
    sar = get_sar(s1_stack_file)
    component_progress_logger.info('29')
    s1_data = read_sar(sar, tile_mask_data_set)
    component_progress_logger.info('43')
    s1_doys = np.array([i.timetuple().tm_yday for i in s1_data.time])
    prior = _get_s1_priors(s1_doys, priors_dir, reprojection, raster_width, raster_height)
    component_progress_logger.info('57')
    sar_inference_data = _get_sar_inference_data(s1_data, s1_doys, priors_dir, reprojection,
                                                 raster_width, raster_height)
    component_progress_logger.info('71')
    lai_outputs, sr_outputs, sm_outputs, \
    Avv_outputs, Bvv_outputs, Cvv_outputs, \
    Avh_outputs, Bvh_outputs, Cvh_outputs, uorbits = do_inversion(sar_inference_data, prior, tile_mask_data_set, False)
    component_progress_logger.info('86')

    times = [i.strftime('%Y-%m-%d') for i in np.array(sar_inference_data.time)[sar_inference_data.time_mask]]

    outfile_names = []
    for parameter_name in parameters:
        for time_step in times:
            outfile_names.append(os.path.join(output_directory, f's1_{parameter_name}_A{time_step}.tif'))
    writer = GeoTiffWriter(outfile_names, tile_mask_data_set.GetGeoTransform(), tile_mask_data_set.GetProjection(),
                           tile_mask_data_set.RasterXSizetile_mask_data_set.RasterYSize,
                           num_bands=None, data_types=None)
    data = []
    for parameter_name in parameters:
        if parameter_name == 'lai':
            for i in range(len(times)):
                data.append(lai_outputs[i, :, :])
        elif parameter_name == 'sr':
            for i in range(len(times)):
                data.append(sr_outputs[i, :, :])
        elif parameter_name == 'sm':
            for i in range(len(times)):
                data.append(sr_outputs[i, :, :])
    writer.write(data, raster_width, raster_height, offset_x, offset_y)
    writer.close()


def _get_interpolated_s2_param(parameter_name: str, priors_dir: str, reprojection: Reprojection, s1_doys: List[int],
                               width: int, height: int):
    formats = [[f'{parameter_name}_A*.tif', 1, 5, 8, '%j'],
               [f'Priors_{parameter_name}_*_global.vrt', 2, 0, 3, '%j'],
               [f's2_{parameter_name}_A*.tif', 2, 1, 11, '%Y-%m-%d']]
    for name_format in formats:
        param_files = glob.glob(os.path.join(priors_dir, name_format[0]))
        if len(param_files) > 0:
            param_data = np.empty((len(param_files), height, width))
            param_doys = np.empty((len(param_files), height, width))
            for i, param_file in enumerate(param_files):
                param_data_set = gdal.Open(param_file)
                reprojected_param_data_set = reprojection.reproject(param_data_set)
                param_data[i] = reprojected_param_data_set.GetRasterBand(1).ReadAsArray()
                date_part = param_file.split('_')[name_format[1]][name_format[2]:name_format[3]]
                if name_format[4] == '%j':
                    param_doys[i] = int(date_part)
                else:
                    param_doys[i] = datetime.strptime(date_part, name_format[4]).timetuple().tm_yday
            return _interpolate_prior(param_doys, s1_doys, param_data), param_doys.min(), param_doys.max(), \
                   np.nanmax(param_data)


def _get_sar_inference_data(s1_data, s1_doys: List[int], priors_dir: str, reprojection: Reprojection,
                            width: int, height: int):
    sar_inference_data = namedtuple('sar_inference_data', 'time lat lon satellite relorbit orbitdirection ang vv vh '
                                                          'lai cab cbrown time_mask fields')
    lai_s1, lai_min_doy, lai_max_doy, lai_nan_max= _get_interpolated_s2_param('lai', priors_dir, reprojection, s1_doys,
                                                                              width, height)
    cab_s1, cab_min_doy, cab_max_doy, cab_nan_max = _get_interpolated_s2_param('cab', priors_dir, reprojection, s1_doys,
                                                                               width, height)
    cb_s1, cb_min_doy, cb_max_doy, cb_nan_max = _get_interpolated_s2_param('cb', priors_dir, reprojection, s1_doys,
                                                                           width, height)

    min_doy = min(lai_min_doy, cab_min_doy, cb_min_doy)
    max_doy = max(lai_max_doy, cab_max_doy, cb_max_doy)

    time_mask = (s1_doys >= min_doy) & (s1_doys <= max_doy)

    patches = sobel(lai_nan_max) > 0.001
    fields = label(patches)[0]
    return sar_inference_data(s1_data.time, s1_data.lat, s1_data.lon, s1_data.satellite, s1_data.relorbit,
                              s1_data.orbitdirection, s1_data.ang, s1_data.vv, s1_data.vh, lai_s1, cab_s1, cb_s1,
                              time_mask, fields)


def _get_s1_priors(s1_doys: List[int], priors_dir: str, reprojection: Reprojection, width: int, height: int):
    prior = namedtuple('prior', 'time sm_prior sm_std sr_prior sr_std')

    soil_moisture_priors = glob.glob(os.path.join(priors_dir, 'sm_*.vrt'))
    soil_roughness_priors = glob.glob(os.path.join(priors_dir, 'sr_*.vrt'))

    sm_priors, sm_stds, sm_doys = _restructure_priors(soil_moisture_priors, reprojection, width, height)
    sr_priors, sr_stds, sr_doys = _restructure_priors(soil_roughness_priors, reprojection, width, height)

    sm_s1 = _interpolate_prior(sm_doys, s1_doys, sm_priors)
    sm_std_s1 = _interpolate_prior(sm_doys, s1_doys, sm_stds)
    sr_s1 = _interpolate_prior(sr_doys, s1_doys, sr_priors)
    sr_std_s1 = _interpolate_prior(sr_doys, s1_doys, sr_stds)

    return prior(s1_doys, sm_s1, sm_std_s1, sr_s1, sr_std_s1)


def _restructure_priors(prior_names: List[str], reprojection: Reprojection, width: int, height: int) -> \
        Tuple[np.array, np.array, np.array]:
    priors = np.empty((len(prior_names), height, width))
    stds = np.empty((len(prior_names), height, width))
    doys = np.empty((len(prior_names), height, width))
    for i, prior_name in enumerate(prior_names):
        date_part = prior_name.split('_')[3].split('.')[0]
        doys[i] = datetime.strptime(date_part, '%Y%m%d').timetuple().tm_yday
        dataset = gdal.Open(prior_name)
        reprojected_dataset = reprojection.reproject(dataset)
        priors[i] = reprojected_dataset.GetRasterBand(1).ReadAsArray()
        stds[i] = reprojected_dataset.GetRasterBand(2).ReadAsArray()
    return priors, stds, doys


def _interpolate_prior(prior_doys: List[int], s1_doys: List[int], prior: np.array) -> np.array:
    f = interp1d(prior_doys, prior, axis=0, bounds_error=False)
    return f(s1_doys)


def _get_mask_data_set_and_reprojection(state_mask: Optional[str] = None, spatial_resolution: Optional[int] = None,
                                        roi: Optional[Union[str, Polygon]] = None, roi_grid: Optional[str] = None,
                                        destination_grid: Optional[str] = None):
    if roi is not None and spatial_resolution is not None:
        if type(roi) is str:
            roi = loads(roi)
        roi_bounds = roi.bounds
        roi_center = roi.centroid
        roi_srs = _get_reference_system(roi_grid)
        destination_srs = _get_reference_system(destination_grid)
        wgs84_srs = _get_reference_system('EPSG:4326')
        if roi_srs is None:
            if destination_srs is None:
                roi_srs = wgs84_srs
                destination_srs = _get_projected_srs(roi_center)
            else:
                roi_srs = destination_srs
        elif destination_srs is None:
            if roi_srs.IsSame(wgs84_srs):
                destination_srs = _get_projected_srs(roi_center)
            else:
                raise ValueError('Cannot derive destination grid for roi grid {}. Please specify destination grid'.
                                 format(roi_grid))
        if state_mask is not None:
            mask_data_set = gdal.Open(state_mask)
        else:
            mask_data_set = _get_default_global_state_mask()
        reprojection = Reprojection(roi_bounds, spatial_resolution, spatial_resolution, destination_srs, roi_srs)
        reprojected_dataset = reprojection.reproject(mask_data_set)
        return reprojected_dataset, reprojection
    elif state_mask is not None:
        state_mask_data_set = gdal.Open(state_mask)
        geo_transform = state_mask_data_set.GetGeoTransform()
        ulx, xres, xskew, uly, yskew, yres = geo_transform
        lrx = ulx + (state_mask_data_set.RasterXSize * xres)
        lry = uly + (state_mask_data_set.RasterYSize * yres)
        roi_bounds = (min(ulx, lrx), min(uly, lry), max(ulx, lrx), max(uly, lry))
        destination_spatial_reference_system = osr.SpatialReference()
        projection = state_mask_data_set.GetProjection()
        destination_spatial_reference_system.ImportFromWkt(projection)
        reprojection = Reprojection(roi_bounds, xres, yres, destination_spatial_reference_system)
        return state_mask_data_set, reprojection
    else:
        raise ValueError("Either state mask or roi and spatial resolution must be given")


def _get_projected_srs(roi_center):
    utm_zone = int(1 + (roi_center.coords[0][0] + 180.0) / 6.0)
    is_northern = int(roi_center.coords[0][1] > 0.0)
    spatial_reference_system = osr.SpatialReference()
    spatial_reference_system.SetWellKnownGeogCS('WGS84')
    spatial_reference_system.SetUTM(utm_zone, is_northern)
    return spatial_reference_system


def _get_default_global_state_mask():
    driver = gdal.GetDriverByName('MEM')
    dataset = driver.Create('', 360, 90, bands=1)
    dataset.SetGeoTransform((-180.0, 1.00, 0.0, 90.0, 0.0, -1.00))
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(np.ones((90, 360)))
    return dataset


def _get_reference_system(wkt: str) -> Optional[osr.SpatialReference]:
    if wkt is None:
        return None
    spatial_reference = osr.SpatialReference()
    if wkt.startswith('EPSG:'):
        epsg_code = int(wkt.split(':')[1])
        spatial_reference.ImportFromEPSG(epsg_code)
    else:
        spatial_reference.ImportFromWkt(wkt)
    return spatial_reference


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MULTIPLY Inference Engine')
    parser.add_argument('-s', "--start_time", help='The start time of the inference period', required=True)
    parser.add_argument("-e", "--end_time", help="The end time of the inference period", required=True)
    parser.add_argument("-i", "--inference_type", help="The type of inference. Must be either 'coarse' or 'high'.",
                        required=True)
    parser.add_argument("-p", "--parameter_list", help="The list of biophysical parameters that shall be derived",
                        required=True)
    parser.add_argument("-pd", "--prior_directory", help="A directory containg the prior files for the "
                                                         "inference period", required=True)
    parser.add_argument("-d", "--datasets_dir", help="The datasets to be used for inferring data.", required=True)
    parser.add_argument("-ps", "--previous_state", help="The directory where the previous state has been saved.")
    parser.add_argument("-ns", "--next_state", help="The directory where the next state shall be saved.")
    parser.add_argument("-em", "--emulators_directory", help="The directory where the emulators are located.")
    parser.add_argument("-fm", "--forward_models", help="The names of the forward models that shall be used.")
    parser.add_argument("-o", "--output_directory", help="The output directory to which the output file shall be "
                                                         "written.", required=True)
    parser.add_argument("-sm", "--state_mask", help="A file containing a state mask to describe the output space "
                                                    "and to mask out pixels. If not given, "
                                                    "Either this or 'roi' and 'spatial_resolution' must be given.")
    parser.add_argument("-roi", "--roi", help="The region of interest describing the area to be retrieved. Not "
                                              "required if 'state_mask' is given.")
    parser.add_argument("-res", "--spatial_resolution", help="The spatial resolution of the destination grid. "
                                                             "Not required if 'state_mask' is given.")
    parser.add_argument("-rg", "--roi_grid", help="A representation of the spatial reference system in which the "
                                                  "roi is given, either as EPSG-code or as WKT representation. "
                                                  "If not given, it is assumed that the roi is given in the "
                                                  "destination spatial reference system.")
    parser.add_argument("-dg", "--destination_grid", help="A representation of the spatial reference system in which "
                                                          "the output shall be given, either as EPSG-code or as WKT "
                                                          "representation. If not given, the output is given in the "
                                                          "grid defined by the 'state_mask'.")
    args = parser.parse_args()
    parameter_list = args.parameter_list.split(',')
    forward_model_list = None
    if args.forward_models is not None:
        forward_model_list = args.forward_models.split(',')
    infer(args.start_time, args.end_time, parameter_list, args.prior_directory, args.datasets_dir, args.previous_state,
          args.next_state, args.output_directory, args.emulators_directory, forward_model_list, args.state_mask,
          args.roi, int(args.spatial_resolution), args.roi_grid, args.destination_grid, False)
