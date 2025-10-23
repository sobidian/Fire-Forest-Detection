import numpy as np 
import cv2 
import glob 
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


feature_vectors = []
labels = []

for i, address in enumerate(glob.glob(r"C:\Users\Sobhan\Desktop\AI_project\fire forest_CNN\*\*\*")):
    img = cv2.imread(address)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0

    feature_vectors.append(img)
    labels.append(address.split("\\")[7])

    if i % 100 == 0:
        print(f"[info] {i}/1500 processed")

feature_vectors = np.array(feature_vectors)

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)
print(labels)

X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.2, random_state=88)


net = models.Sequential([
                        layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
                        layers.MaxPooling2D(),
                        layers.Conv2D(32, (3,3), activation="relu"),
                        layers.MaxPooling2D(),
                        layers.Flatten(),
                        layers.Dense(100, activation="relu"),
                        layers.Dense(40, activation="relu"),
                        layers.Dense(2, activation="softmax")  
])

print(net.summary())

net.compile(optimizer="SGD",
            loss="categorical_crossentropy",
            metrics=["accuracy"])

H = net.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test), epochs=25)



# net.save(r"C:\Users\Sobhan\Desktop\AI_project\fire forest_CNN\save\CNN.h5")


#plot
plt.style.use("ggplot")
plt.plot(H.history["accuracy"], label = "train")
plt.plot(H.history["val_accuracy"], label = "test")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Fire/Non Fire")
plt.show()
