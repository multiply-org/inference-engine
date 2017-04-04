import unittest
import dummy_inference_engine

class dummy_inference_engine_test(unittest.TestCase):

    def test_assimilate(self):
        engine = dummy_inference_engine.dummy_inference_engine()

        coarse_res_brdf_descriptors= []
        high_res_sdr = []
        grd_sar_data = []
        prior = object
        optical_forward_operator_emulator = object
        sar_forward_operator_emulator = object

        high_res_biophysical_params = engine.infer(coarse_res_brdf_descriptors, high_res_sdr, grd_sar_data, prior,
                             optical_forward_operator_emulator, sar_forward_operator_emulator)
