import yaml

from classifier.inference import InferencePipeline


def test_inference():

    config = yaml.safe_load(open("config/config.yml"))

    config["model"]["output_model_onnx"] = "tests/model.onnx"

    pipeline = InferencePipeline(config)
    text = "How did serfdom develop in and then leave Russia ?"
    predicted_class = pipeline.run_onnx_session(text)
    assert predicted_class is not None
