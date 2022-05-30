from PIL import Image
from utils import *
import pickle
import os
import time
from flask import Flask, render_template, request, redirect

"""
Load vectors and path
"""
vectors = pickle.load(open("./results/vectors.pkl","rb"))
paths = pickle.load(open("./results/paths.pkl","rb"))

def clearUpload(path):
    data = os.listdir(path)
    if len(data) > 0:
        for img in data:
            os.remove(os.path.join(path, img))

app = Flask(__name__, template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_DIRECTORY'] = './uploads'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# app.config['ALLOWED_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.gif']

"""No Cache loading
    ref: https://stackoverflow.com/questions/45583828/python-flask-not-updating-images
"""
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    clearUpload('./uploads')
    file = request.files['file']
    if file:
        file.save(os.path.join(app.config['UPLOAD_DIRECTORY'], file.filename))
        return redirect('/query')
    else:
        print("No file selected!")
        return redirect('/')

@app.route('/query')
def query():
    img_query = os.path.join(app.config['UPLOAD_DIRECTORY'], os.listdir(app.config['UPLOAD_DIRECTORY'])[0])
    model = get_extract_model()
    # Query image features extraction
    search_vector = extract_vector(model, img_query)

    # Distance from query's vector to all vector in dataset
    distance = np.linalg.norm(vectors - search_vector, axis=1) # L2-Norm

    K = 10 # Return top K image same as query image 
    ids = np.argsort(distance)[:K]

    nearest_image = [paths[id] for id in ids]
    query_ans = []

    for path in nearest_image:
        tmp = path.split('/')
        version = f'?v={int(round(time.time() * 1000))}'
        full_path = os.path.join('static/dataset', tmp[2], tmp[3]) + version
        query_ans.append(full_path)
    
    return render_template('home.html', query_img = query_ans)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port='8080',debug=True)