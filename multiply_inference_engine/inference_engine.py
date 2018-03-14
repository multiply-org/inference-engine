import numpy as np

from datetime import datetime
from kafka import LinearKalman
from kafka.inference import create_prosail_observation_operator
from multiply_core import util
from shapely.geometry import Polygon
from shapely.wkt import loads
from typing import List, Optional, Union

__author__ = "Tonio Fincke (Brockmann Consult GmbH)"


# class InferenceEngine(object):

def infer(start_time: Union[str, datetime], end_time: Union[str, datetime],
          dataset_urls: List[str], output_format: str, parameter_list: List[str],
          roi: Union[str, Polygon], roi_grid: Optional[str], destination_grid: Optional[str]="EPSG:4326"):
    """

    :param start_time:
    :param end_time:
    :param dataset_urls:
    :param output_format:
    :param parameter_list: The list of bio-physical parameters that shall be inferred.
    :param roi: The region of interest, either as shapely Polygon or in a WKT representation.
    :param roi_grid: A representation of the spatial reference system in which the roi is given, either as EPSG-code
    or as WKT representation. If not given, it is assumed that the roi is given in the
    destination spatial reference system.
    :param destination_grid: A representation of the spatial reference system in which the output shall be given,
    either as EPSG-code or as WKT representation. If not given, the output is given in WGS84 coordinates.
    :return:
    """
    #TODO consider not using WGS84 as default, but picking the correct UTM zones
    # we assume that time is derived for one time step; or, to be more precise, for one time period (with no
    # intermediate time steps). This time step/time period is described by start time and end time.
    if start_time is str:
        start_time = util.get_time_from_string(start_time)
    if end_time is str:
        end_time = util.get_time_from_string(end_time)
    if roi is str:
        roi = loads(roi)
    # bounds: minx, miny, maxx, maxy
    roi_bounds = roi.bounds
    # s2_observations: Observations object according to the Kafka Observations interface
    # class to output the state, has dump_data
    # mask: 1D(?) numpy array, boolean
    # create_prosail_observation_operator: From Kafka
    # parameter_list: The list of biophysical parameters we are interested in
    # prior: from inference prior
    # kf = LinearKalman(s2_observations, output, mask, create_prosail_observation_operator, parameter_list,
    #                   state_propagation=None, prior=the_prior)
    # x_forecast <= from prior

    # what is this?
    # Q = np.zeros_like(x_forecast)
    # Q[6::7] = 0.025

    # kf.set_trajectory_model()
    # kf.set_trajectory_uncertainty(Q)

    # set up the time grid. We only have two dates, as the temporal stepping is done in the orchestrator.
    time_grid = [start_time, end_time]

    # i do not know what the ones in the bottom half mean
    # kf.run(time_grid=time_grid, x_forecast=x_forecast, P_forecast=None, P_forecast_inverse=P_forecast_inv,
    #        diag_str="diagnostics", approx_diagonal=True, refine_diag=True, is_robust=False, iter_obs_op=True)

# def
