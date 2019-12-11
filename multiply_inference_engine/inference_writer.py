import numpy as np
import os
import scipy.sparse as sp

from datetime import datetime
from multiply_core.observations import GeoTiffWriter
from typing import List, Optional

__author__ = "Tonio Fincke (Brockmann Consult GmbH)"


class InferenceWriter:

    def __init__(self, actual_parameter_list: List[str], full_parameter_list: List[str], output_folder: str,
                 time_step: datetime, geo_transform: tuple, projection: str, width: int, height: int,
                 output_format: str = "GeoTiff", state_folder: str = None):
        self._actual_parameter_list = actual_parameter_list
        file_names = []
        self.param_positions = {}
        num_bands = []
        for param in actual_parameter_list:
            if param in full_parameter_list:
                file_names.append(os.path.join(output_folder, "%s_%s.tif" % (param, time_step.strftime("A%Y%j"))))
                num_bands.append(1)
                file_names.append(os.path.join(output_folder, "%s_%s_unc.tif" % (param, time_step.strftime("A%Y%j"))))
                num_bands.append(1)
                self.param_positions[param] = full_parameter_list.index(param)
        if output_format == "GeoTiff":
            self.writer = GeoTiffWriter(file_names, geo_transform, projection, width, height, num_bands,
                                        data_types=None)
        else:
            raise ValueError(f"Format {output_format} is not supported.")
        self._width = width
        self._height = height
        self._offset_x = 0
        self._offset_y = 0
        self.state_folder = state_folder

    def set_tile(self, width: int, height: int, offset_x: int, offset_y: int):
        self._width = width
        self._height = height
        self._offset_x = offset_x
        self._offset_y = offset_y

    def dump_data(self, time_step: Optional[datetime], x_analysis: np.array, p_analysis: np.array,
                  p_analysis_inv: sp.coo_matrix, state_mask: np.array, n_params: int):
        data = []
        for param in self.param_positions:
            index = self.param_positions[param]
            param_values = np.zeros(state_mask.shape, dtype=np.float32)
            param_values[state_mask] = x_analysis[index::n_params]
            data.append(param_values)
            param_unc = np.zeros(state_mask.shape, dtype=np.float32)
            param_unc[state_mask] = 1. / np.sqrt(p_analysis_inv.diagonal()[index::n_params])
            data.append(param_unc)
        self.writer.write(data, self._width, self._height, self._offset_x, self._offset_y)

    def dump_state(self, timestep: datetime, x_analysis: np.array, p_analysis: np.array, p_analysis_inv: np.array,
                   state_mask: np.array):
        if self.state_folder is not None and os.path.exists(self.state_folder):
            # Dump to disk P_analysis_inv as sparse matrix in npz
            if p_analysis_inv is not None:
                file_name = os.path.join(self.state_folder, "P_analysis_inv_%s.npz" % (timestep.strftime("A%Y%j")))
                try:
                    sp.save_npz(file_name, p_analysis_inv)
                except:
                    if os.path.exists(file_name):
                        os.remove(file_name)
            if p_analysis is not None:
                file_name = os.path.join(self.state_folder, "P_analysis_%s.npz" % (timestep.strftime("A%Y%j")))
                np.savez(file_name, p_analysis)

            # Dump as well the whole x_analysis in a single vector
            file_name = os.path.join(self.state_folder, "x_analysis_%s.npz" %
                                     (timestep.strftime("A%Y%j")))
            np.savez(file_name, x_analysis)

            # ... and the state mask ...
            file_name = os.path.join(self.state_folder, "state_mask_%s.npz" %
                                     (timestep.strftime("A%Y%j")))
            np.savez(file_name, state_mask)

    def close(self):
        self.writer.close()
