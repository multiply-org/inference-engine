import argparse
import gdal
import glob
import logging
import numpy as np
import os
import osr
import scipy.sparse as sp

from multiply_inference_engine.inference_prior import InferencePrior

from datetime import datetime
from kafka import LinearKalman
from kafka.input_output import KafkaOutput
from kafka.inference import create_prosail_observation_operator
from kafka.inference.narrowbandSAIL_tools import SAILPrior
from kafka.inference.narrowbandSAIL_tools import propagate_LAI_narrowbandSAIL as propagator
from multiply_core.observations import data_validation, ObservationsFactory
from multiply_core.util import FileRef, FileRefCreation, Reprojection, get_time_from_string
from shapely.geometry import Polygon
from shapely.wkt import loads
from typing import List, Optional, Union

__author__ = "Tonio Fincke (Brockmann Consult GmbH)"


def _get_valid_files(datasets_dir: str) -> FileRef:
    file_refs = []
    file_ref_creation = FileRefCreation()
    found_files = glob.glob(datasets_dir + '/**', recursive=True)
    for found_file in found_files:
        type = data_validation.get_valid_type(found_file)
        if type is not '':
            file_ref = file_ref_creation.get_file_ref(type, found_file)
            if file_ref is not None:
                logging.getLogger('inference_engine').info('retrieve observations from {}'.format(file_ref.url))
                file_refs.append(file_ref)
    return file_refs


def infer(start_time: Union[str, datetime],
          end_time: Union[str, datetime],
          inference_type: str,
          parameter_list: List[str],
          prior_directory: str,
          datasets_dir: str,
          previous_state_dir: str,
          next_state_dir: str,
          emulators_directory: str,
          output_directory: str,
          state_mask: Optional[str],
          roi: Optional[Union[str, Polygon]],
          spatial_resolution: Optional[int],
          roi_grid: Optional[str],
          destination_grid: Optional[str]="EPSG:4326"):
    """
    :param start_time: The start time of the inference period
    :param end_time: The end time of the inference period
    :param inference_type: The type of inference. Must be either 'coarse' or 'high'.
    :param parameter_list: The list of bio-physical parameters that shall be inferred.
    :param prior_directory: A directory where the global .vrt-files of the priors are located.
    :param datasets_dir: A directory where the input data is located.
    :param previous_state_dir: A directory where the state from the previous inference period has been saved to.
    :param next_state_dir: A directory to which the state can be saved.
    :param emulators_directory: The directory where the emulators are placed.
    :param output_directory: The directory to which the output shall be written.
    :param state_mask: A file that defines both the region of interest and the destination grid to which the output
    shall be written. If not given, roi and spatial_resolution must be set.
    :param roi: The region of interest, either as shapely Polygon or in a WKT representation.
    :param spatial_resolution: The spatial resolution of the destination grid.
    :param roi_grid: A representation of the spatial reference system in which the roi is given, either as EPSG-code
    or as WKT representation. If not given, it is assumed that the roi is given in the
    destination spatial reference system.
    :param destination_grid: A representation of the spatial reference system in which the output shall be given,
    either as EPSG-code or as WKT representation. If not given, the output is given in WGS84 coordinates.
    :return:
    """
    # TODO use actually passed parameter list! this one is sail model specific
    parameter_list = ['n', 'cab', 'car', 'cbrown', 'cw', 'cm', 'lai', 'ala', 'bsoil', 'psoil']
    # we assume that time is derived for one time step; or, to be more precise, for one time period (with no
    # intermediate time steps). This time step/time period is described by start time and end time.
    if type(start_time) is str:
        start_time = get_time_from_string(start_time)
    if type(end_time) is str:
        end_time = get_time_from_string(end_time)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    prior_reference_dataset = None
    reprojection = None
    output = None
    mask = None
    inference_prior = None
    if state_mask is not None:
        state_mask_data_set = gdal.Open(state_mask)
        prior_reference_dataset = state_mask_data_set
        geo_transform = state_mask_data_set.GetGeoTransform()
        ulx, xres, xskew, uly, yskew, yres = geo_transform
        lrx = ulx + (state_mask_data_set.RasterXSize * xres)
        lry = uly + (state_mask_data_set.RasterYSize * yres)
        roi_bounds = (min(ulx, lrx), min(uly, lry), max(ulx, lrx), max(uly, lry))
        destination_spatial_reference_system = osr.SpatialReference()
        projection = state_mask_data_set.GetProjection()
        destination_spatial_reference_system.ImportFromWkt(projection)
        reprojection = Reprojection(roi_bounds, xres, yres, destination_spatial_reference_system)
        output = KafkaOutput(parameter_list, geo_transform, projection, output_directory, next_state_dir)
        mask = state_mask_data_set.ReadAsArray().astype(np.bool)
        inference_prior = SAILPrior(parameter_list, state_mask)
    else:
        # TODO interprete destination and bounds grid correctly
        if roi is str:
            roi = loads(roi)
        # bounds: minx, miny, maxx, maxy
        roi_bounds = roi.bounds
        # reprojection = Reprojection(roi_bounds, spatial_resolution, spatial_resolution, )

    # prior_files = glob.glob(prior_directory + '/*.vrt')
    # inference_prior = InferencePrior('', global_prior_files=prior_files, reference_dataset=prior_reference_dataset)



    file_refs = _get_valid_files(datasets_dir)
    observations_factory = ObservationsFactory()
    observations_factory.sort_file_ref_list(file_refs)
    # an observations wrapper to be passed to kafka
    observations = observations_factory.create_observations(file_refs, reprojection, emulators_directory)

    linear_kalman = LinearKalman(observations, output, mask, create_prosail_observation_operator, parameter_list,
                      state_propagation=propagator, prior=None, linear=False)

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
    if p_forecast_inv is None or x_forecast is None:
        # processed_prior = inference_prior.process_prior(parameter_list, start_time, mask)
        processed_prior = inference_prior.process_prior(None)
        if x_forecast is None:
            x_forecast = processed_prior[0]
        if p_forecast_inv is None:
            p_forecast_inv = processed_prior[1]

    # Inflation amount for propagation
    Q = np.zeros_like(x_forecast)
    Q[6::10] = 0.05

    linear_kalman.set_trajectory_model()
    linear_kalman.set_trajectory_uncertainty(Q)

    time_grid = [start_time, end_time]
    print(x_forecast)
    linear_kalman.run(time_grid, x_forecast, None, p_forecast_inv, iter_obs_op=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MULTIPLY Inference Engine')
    parser.add_argument('-s', "--start_time", help='The start time of the inference period',required=True)
    parser.add_argument("-e", "--end_time", help="The end time of the inference period", required=True)
    parser.add_argument("-i", "--inference_type",help="The type of inference. Must be either 'coarse' or 'high'.",
                        required=True)
    parser.add_argument("-p", "--parameter_list",help="The list of biophysical parameters that shall be derived",
                        required=True)
    parser.add_argument("-pd", "--prior_directory", help="A directory containg the prior files for the "
                                                         "inference period", required=True)
    parser.add_argument("-d", "--datasets_dir", help="The datasets to be used for inferring data.", required=True)
    parser.add_argument("-ps", "--previous_state", help="The directory where the previous state has been saved.")
    parser.add_argument("-ns", "--next_state", help="The directory where the next state shall be saved.")
    parser.add_argument("-em", "--emulators_directory", help="The directory where the emulators are located.")
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
                                                          "grid defined by the 'state_mask'. If no 'state_mask is "
                                                          "given,' the output is written in WGS84 coordinates.",
                        default='EPSG:4326')
    args = parser.parse_args()
    parameter_list = args.parameter_list.split(',')
    infer(args.start_time, args.end_time, args.inference_type, parameter_list, args.prior_directory,
          args.datasets_dir, args.previous_state, args.next_state, args.emulators_directory, args.output_directory,
          args.state_mask, args.roi, args.spatial_resolution, args.roi_grid, args.destination_grid)
