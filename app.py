from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

classes = ['10000','100000','20000', '200000', '50000', '500000']

model = load_model('final_model.h5')

# Dự đoán nhãn
def predict_label(image_path):
    i = image.load_img(image_path, target_size=(224,224))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1,224,224,3)
    pred = model.predict(i)
    return classes[np.argmax(pred[0])]

#route
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path) # lưu đường dẫn để dự đoán nhãn của ảnh

        p = predict_label(img_path)

        return render_template("index.html", img_path = img_path, prediction = p)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port='6868')
