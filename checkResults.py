import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('digits_model_new.h5')

for i in range(1,6):
    img = cv.imread(f'{i}.png')[:,:,0]
    img = np.invert(np.array([img]))
    pred = model.predict(img)
    print(f'The result is probably: {np.argmax(pred)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
