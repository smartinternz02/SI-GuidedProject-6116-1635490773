# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:24:55 2021

@author: rajde
"""




from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np

model=load_model(r"D:\project_smartbridge\Malaria_model.h5")
img=image.load_img(r"D:\project_smartbridge\cell_images\test\Uninfected\u1.png", target_size=(128,128))
x=image.img_to_array(img)
#print(x)
#print(x.shape)

x=np.expand_dims(x, axis=0)
#print(x.shape)

#pred=model.predict_classes(x)
y=model.predict(x)
pred= np.argmax(y, axis=1)
print(y)


'''if (y==0):
    print('Parasitized')
else:
    print("Uninfected")
'''

index=['Parasitized','Uninfected']
result=str(index[pred[0]])
print(result)