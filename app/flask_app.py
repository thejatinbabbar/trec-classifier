from flask import Flask, request, jsonify
from classifier.inference import InferencePipeline

app = Flask(__name__)

# Load the model (choose between PyTorch and ONNX)
model_type = "pytorch"  # or "onnx"
checkpoint_path = "model.pth"
onnx_model_path = "model.onnx"

inference_pipeline = InferencePipeline(model_type, checkpoint_path)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    prediction = inference_pipeline.predict(text)
    return jsonify({"prediction": prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
