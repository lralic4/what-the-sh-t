from flask import Flask, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
img_height = 180
img_width = 180
class_names = ['type-1', 'type-2', 'type-3', 'type-4', 'type-5', 'type-6', 'type-7']

@app.route('/bristol-chart', methods=['GET'])
def get_bristol_chart_classification():
    model = tf.keras.models.load_model('../bristol-model/bristol-model.h5')


    fullPath = os.path.abspath("../bristol-model/testimage1.jpeg")  # or similar, depending on your scenario
    image_path = tf.keras.utils.get_file('testimage1.jpeg', 'file://'+fullPath)

    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    return jsonify({'classification': class_names[np.argmax(score)], 'confidence': 100 * np.max(score)})

if __name__ == '__main__':
    app.run()
