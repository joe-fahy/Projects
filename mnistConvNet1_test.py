import cv2
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
#import pillow
import numpy as np
import cv2
from matplotlib import pyplot as plt


print("This is a script to classify handwritten digits.")

model = load_model('mnistCnnModel_1.h5')



#img = cv2.imread("abc.tiff", mode='RGB')
#img = cv2.imread('0.jpg', 1)
#img_as_np = np.asarray(img)

img = cv2.imread('0.jpeg',0)
img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
#img.resize(28,28)
plt.imshow(img, cmap='gray')
plt.show()

#plt.imshow(res, cmap='gray')
#plt.show()
img_width = 28
img_height = 28

#img = load_img('0.jpg',False,target_size=(img_width,img_height))
#plt.figure(figsize=(8,12))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = model.predict_classes(x)
prob = model.predict_proba(x)
print(preds, prob)


#print(type(img))
#classes = model.predict(img)
#print(classes)
