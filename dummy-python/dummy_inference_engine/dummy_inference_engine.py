class dummy_inference_engine:

    def infer(self, coarse_res_brdf_descriptors, high_res_sdr, grd_sar_data, prior, optical_forward_operator_emulator,
              sar_forward_operator_emulator):
        print("The inference engine infers high resolution biophysical parameters.")
        high_res_biophysical_parameters = []
        return high_res_biophysical_parameters