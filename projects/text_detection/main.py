import os
import cv2
import matplotlib.pyplot as plt
from .preprocessing import preprocessing
image = cv2.imread(os.path.join('.', 'projects/text_detection/page_01.jpg'))
if image is None:
    print("Failed to load image. Please check the file path.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocessing(image_rgb)
    plt.imshow(image)
    plt.axis('off')  # Tắt hiển thị trục
    plt.show()