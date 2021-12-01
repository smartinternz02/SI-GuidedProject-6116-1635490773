# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:29:23 2021

@author: rajde
"""


import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template
#from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = load_model("Malaria_model.h5")
                 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (128,128)) 
        x = image.img_to_array(img)
        print(x)
        x = np.expand_dims(x,axis =0)
        print(x)

       # preds = model.predict_classes(x)
        y=model.predict(x)
        preds=np.argmax(y,axis=1)
       
        print("prediction",preds)
        index = ['Parasitized','Uninfected']
        text = "The classified Malaria is : " + str(index[preds[0]])
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False,port=8000)
