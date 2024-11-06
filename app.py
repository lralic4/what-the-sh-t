from flask import Flask, render_template, request, redirect
import tensorflow as tf
import numpy as np
import os
import uuid

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
        print('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        print('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        print('Saving file - ', file.filename)

        ''' 
        This version of tensorflow caches files -
        need to use different file name to avoid cache
        '''
        filename = str(uuid.uuid4())
        file.save(f'./{filename}.jpeg') 

        return filename

@app.route('/bristol-chart', methods=['POST'])
def get_bristol_chart_classification():
    filename = save_image(request)

    fullPath = os.path.abspath(f'./{filename}.jpeg')  # or similar, depending on your scenario
    image_path = tf.keras.utils.get_file(f'{filename}.jpeg', 'file://'+fullPath)

    img = tf.keras.utils.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = MODEL.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    os.remove(f'./{filename}.jpeg')

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
