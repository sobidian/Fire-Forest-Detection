import cv2
import numpy as np
from tensorflow.keras import models

net = models.load_model(r"save\CNN.h5")

img= cv2.imread(r"C:\Users\Sobhan\Desktop\AI_project\fire forest_CNN\Test\fire_0537.jpg")

img = cv2.resize(img, (32, 32)) 
img = img / 255.0

img = np.array([img])
output = net.predict(img)[0]
max_output = np.argmax(output)

category_name = ["Fire" ," Non Fire"]
text = category_name[max_output]

print(text)