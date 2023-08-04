from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from predict import predict

app = Flask(__name__)


# 处理函数，接收图像并返回检测结果
@app.route("/predict", methods=["POST"])
def detect_objects():
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400

    image_file = request.files["image"]
    image_file = image_file.read()
    image = Image.open(BytesIO(image_file))

    result = predict(image)  # train the model

    result_list = []
    for box in result[0].boxes.data:
        new_box = [int(num) for num in box]
        result_list.append(new_box)

    data = {"status": "success", "result": result_list}

    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
