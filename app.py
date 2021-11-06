from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model

app = Flask(__name__)

classes = ['0','10000','20000','50000', '100000', '200000', '500000']

def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # Dong bang cac layer
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Tao model
    input = Input(shape=(224, 224, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    # Them cac layer FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(7, activation='softmax', name='predictions')(x)

    # Compile
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

model = get_model()

model.load_weights('weights-14-0.93.hdf5')

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

        return render_template("index.html", img = img_path, prediction = p)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port='6868')
