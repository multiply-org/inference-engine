from multiply_inference_engine import InferenceEngine

__author__ = "Tonio Fincke (Brockmann Consult GmbH)"

PATH_TO_EMULATOR = './test_data/dummy_emulator.pkl'


def test_infer():
    inference_engine = InferenceEngine()
    inference_engine.infer()
