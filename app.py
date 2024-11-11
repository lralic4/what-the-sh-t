from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import transforms
from PIL import Image
import os
import uuid

app = Flask(__name__)

IMG_HEIGHT = 180
IMG_WIDTH = 180
ALLOWED_EXTENSIONS = {'jpeg'}
MODEL = model = torch.load('./bristol-model/model.pth', weights_only=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

BRISTOL_STOOLS = [
    {'name': 'Type 1', 'title': 'Separate Hard Lumps', 'description': 'If your poop resembles separate hard lumps, it falls under Type 1 on the Bristol Stool Chart. This shape indicates constipation and suggests that your stool is spending too much time in the colon, causing excessive water absorption. It may be a sign of inadequate fiber intake, dehydration, or certain medications.', 'img_link': 'type-1.png'},
    {'name': 'Type 2', 'title': 'Lumpy and Sausage-Like', 'description': 'Type 2 stool is characterized by lumpy and sausage-like appearance. While it indicates mild constipation, it is closer to the ideal poop shape. This suggests that you may need to increase your fiber intake, drink more water, and ensure regular physical activity to promote better bowel movements.', 'img_link': 'type-2.png'},
    {'name': 'Type 3', 'title': 'Sausage-Like with Cracks', 'description': 'Type 3 stool is sausage-like with cracks on its surface. It indicates normal bowel movements and signifies a healthy balance of water content and stool formation. This shape represents a healthy poop consistency that is relatively easy to pass.', 'img_link': 'type-3.png'},
    {'name': 'Type 4', 'title': 'Smooth and Sausage-Like', 'description': 'Type 4 stool, often considered the “Goldilocks” of poop shapes, is smooth and sausage-like, resembling a well-formed snake. It indicates a healthy balance of fiber, water, and transit time through the colon. Type 4 is the ideal poop shape, suggesting a healthy digestive system and regular bowel movements.', 'img_link': 'type-4.png'}, 
    {'name': 'Type 5', 'title': 'Soft Blobs with Clear-Cut Edges', 'description': 'Type 5 stool appears as soft blobs with clear-cut edges. It is looser than Type 4 but still maintains some form. This shape may indicate a slightly accelerated transit time through the colon, possibly due to a high-fiber diet or certain medications.', 'img_link': 'type-5.png'},
    {'name': 'Type 6', 'title': 'Fluffy and Mashed Potatoes-Like', 'description': 'Type 6 stool has a fluffy and mashed potatoes-like consistency. It suggests looser stools and may be associated with conditions such as irritable bowel syndrome (IBS), dietary factors, or certain medications. If you consistently experience Type 6 stools without any specific reason, it is advisable to seek medical evaluation.', 'img_link': 'type-6.png'},
    {'name': 'Type 7', 'title': 'Watery and Entirely Liquid', 'description': 'Type 7 stool is entirely liquid, resembling watery diarrhea. This shape indicates severe diarrhea and suggests an increased fluid content in the stool. It can be a result of infections, food poisoning, medications, or gastrointestinal disorders. If you experience persistent Type 7 stools, it is crucial to seek medical attention for proper diagnosis and treatment.', 'img_link': 'type-7.png'},
]

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
 
        # Use unique file name to prevent caching file
        filename = str(uuid.uuid4())
        file.save(f'./{filename}.jpeg') 

        return filename

@app.route('/bristol-chart', methods=['POST'])
def get_bristol_chart_classification():
    filename = save_image(request)

    fullPath = os.path.abspath(f'./{filename}.jpeg')  # or similar, depending on your scenario

    img = Image.open(fullPath)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, _ = torch.max(probs, 1)

    os.remove(f'./{filename}.jpeg')
    
    index = preds[0]
    confidence = conf.item()

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(BRISTOL_STOOLS[index]['name'], 100 * confidence)
    )

    context = {
        'name': BRISTOL_STOOLS[index]['name'],
        'title': BRISTOL_STOOLS[index]['title'],
        'description': BRISTOL_STOOLS[index]['description'],
        'img_link': BRISTOL_STOOLS[index]['img_link'],
        'confidence': round(100 * confidence, 2)
    }

    return render_template('results.html', **context)

if __name__ == '__main__':
    app.run(port=3000)
