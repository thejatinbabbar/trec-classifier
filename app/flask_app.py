import yaml
from flask import Flask, request, jsonify
from classifier.inference import InferencePipeline


app = Flask(__name__)

config = yaml.safe_load(open('config/config.yml'))
inference_pipeline = InferencePipeline(config)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    prediction = inference_pipeline.run_onnx_session(text)
    return jsonify({"prediction": prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
