from classifier.inference import InferencePipeline


def test_pytorch_inference():
    pipeline = InferencePipeline(model_type="pytorch", checkpoint_path="best-checkpoint.ckpt")
    text = "What is the capital of France?"
    predicted_class = pipeline.predict(text)
    assert predicted_class is not None

def test_onnx_inference():
    pipeline = InferencePipeline(model_type="onnx", onnx_model_path="model.onnx")
    text = "What is the capital of France?"
    predicted_class = pipeline.predict(text)
    assert predicted_class is not None
