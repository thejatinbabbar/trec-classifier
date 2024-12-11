from app.inference import InferencePipeline


def test_inference(config):

    pipeline = InferencePipeline(config)
    text = "How did serfdom develop in and then leave Russia ?"
    predicted_class = pipeline.run_onnx_session(text)
    assert predicted_class is not None
