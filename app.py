from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import transforms
from PIL import Image
import os
import uuid

app = Flask(__name__)

IMG_HEIGHT = 180
IMG_WIDTH = 180
ALLOWED_EXTENSIONS = {'jpeg', 'png', 'jpg'}
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
    {'name': 'Type 1', 'title': 'Separate hard lumps', 'description': 'If you have these hard, separate pellets of poop that are hard to pass, it is a sign of severe constipation. That means your poop is taking a long time to get through your digestive system. When that happens, more of the water that would otherwise be part of your poop is absorbed by your colon, leaving dryer, smaller stools. The most common causes are getting too little fluid and fiber in your diet. Medications, stress, and illness can also play roles.', 'img_link': 'type-1.png'},
    {'name': 'Type 2', 'title': 'Sausage-shaped, but lumpy', 'description': 'These hard, lumpy poops are a little bigger, but still signal constipation and often result from a lack of fluid and fiber. In addition to changing your diet, you might try more exercise to move things along.', 'img_link': 'type-2.png'},
    {'name': 'Type 3', 'title': 'Sausage-shaped, but with cracks on surface', 'description': 'A sausage-shaped poop with cracks on the surface is considered normal on the Bristol Stool Scale.', 'img_link': 'type-3.png'},
    {'name': 'Type 4', 'title': 'Sausage or snake-like, smooth and soft', 'description': 'A thinner, more snake-like poop that is smooth and soft is also considered normal under Bristol criteria.', 'img_link': 'type-4.png'}, 
    {'name': 'Type 5', 'title': 'Soft nlobs with clear-cut edges', 'description': 'If you are producing soft blobs of poop with clear edges, you are tending toward diarrhea. But you might be surprised to learn that the problem could be eating too little fiber, as fiber helps firm up your poop.', 'img_link': 'type-5.png'},
    {'name': 'Type 6', 'title': 'Fluffy pieces with ragged edges, Mushy', 'description': 'If you are passing fluffy, mushy, pieces of poop with ragged edges, that is diarrhea. It is a sign that your food is making a rapid trip through your digestive tract, giving your colon too little time to absorb fluid and form firmer stools. You could have a viral infection such as norovirus, food poisoning, or some other digestive issue. Stress can also play a role.', 'img_link': 'type-6.png'},
    {'name': 'Type 7', 'title': 'Watery, no solid pieces', 'description': 'Watery stools with no solid pieces are symptoms of severe diarrhea. Most diarrhea lasts a day or two and then goes away on its own. But if you have this kind of poop or type 6 stool a lot of the time, talk to your doctor. Chronic diarrhea can be a symptom of conditions that cause irritation or inflammation of the bowels, including irritable bowel syndrome, Crohns disease, and ulcerative colitis.', 'img_link': 'type-7.png'},
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
        filename, ext = os.path.splitext(file.filename)
 
        # Use unique file name to prevent caching file
        filename = str(uuid.uuid4())
        file.save(f'./{filename}{ext}') 

        return filename, ext

@app.route('/bristol-chart', methods=['POST'])
def get_bristol_chart_classification():
    filename, ext = save_image(request)

    fullPath = os.path.abspath(f'./{filename}{ext}')  # or similar, depending on your scenario

    img = Image.open(fullPath)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, _ = torch.max(probs, 1)

    os.remove(f'./{filename}{ext}')
    
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
    app.run()
