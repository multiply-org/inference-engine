import argparse
import numpy as np

from datetime import datetime
from kafka import LinearKalman
from kafka.inference import create_prosail_observation_operator
from multiply_core import util
from shapely.geometry import Polygon
from shapely.wkt import loads
from typing import List, Optional, Union

__author__ = "Tonio Fincke (Brockmann Consult GmbH)"


def infer(start_time: Union[str, datetime],
          end_time: Union[str, datetime],
          inference_type: str,
          parameter_list: List[str],
          prior_files: List[str],
          dataset_urls: List[str],
          state_mask: Optional[Union[str, np.array]],
          roi: Optional[Union[str, Polygon]],
          spatial_resolution: Optional[float],
          roi_grid: Optional[str],
          destination_grid: Optional[str]="EPSG:4326"):
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
    #todo add something here so that it is possible to use custom stepping (e.g., a two weeks temporal resolution)
    time_grid = [start_time, end_time]

    # i do not know what the ones in the bottom half mean
    # kf.run(time_grid=time_grid, x_forecast=x_forecast, P_forecast=None, P_forecast_inverse=P_forecast_inv,
    #        diag_str="diagnostics", approx_diagonal=True, refine_diag=True, is_robust=False, iter_obs_op=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MULTIPLY Inference Engine')
    parser.add_argument('-s', "--start_time", help='The start time of the inference period',required=True)
    parser.add_argument("-e", "--end_time", help="The end time of the inference period", required=True)
    parser.add_argument("-i", "--inference_type",help="The type of inference. Must be either 'coarse' or 'high'.",
                        required=True)
    parser.add_argument("-p", "--parameter_list",help="The list of biophysical parameters that shall be derived",
                        required=True)
    parser.add_argument("-pf", "--prior_files", help="The prior files for the inference period", required=True)
    parser.add_argument("-d", "--dataset_urls", help="The datasets to be used for inferring data.", required=True)
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
    infer(args.start_time, args.end_time, args.inference_type, args.parameter_list, args.prior_files, args.dataset_urls,
          args.state_mask, args.roi, args.spatial_resolution, args.roi_grid, args.destination_grid)
