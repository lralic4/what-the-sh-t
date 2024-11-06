from flask import Flask, jsonify, render_template, request, redirect, url_for, flash
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
IMG_HEIGHT = 180
IMG_WIDTH = 180
CLASS_NAMES = ['type-1', 'type-2', 'type-3', 'type-4', 'type-5', 'type-6', 'type-7']
ALLOWED_EXTENSIONS = {'jpeg'}
MODEL = tf.keras.models.load_model('./bristol-model/bristol-model.h5')

@app.route('/')
def home():
   return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_image(request):

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        file.save('./temp.jpeg')  

@app.route('/bristol-chart', methods=['POST'])
def get_bristol_chart_classification():
    save_image(request)

    fullPath = os.path.abspath("./temp.jpeg")  # or similar, depending on your scenario
    image_path = tf.keras.utils.get_file('temp.jpeg', 'file://'+fullPath)

    img = tf.keras.utils.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = MODEL.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(CLASS_NAMES[np.argmax(score)], 100 * np.max(score))
    )

    context = {
        'class': CLASS_NAMES[np.argmax(score)],
        'confidence': 100 * np.max(score)
    }

    return render_template('results.html', **context)

if __name__ == '__main__':
    app.run()
