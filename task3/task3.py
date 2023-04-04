import cv2
import numpy as np
import matplotlib.pyplot as plt

train_image = cv2.imread("./train_images/011-trash.png")

train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax.imshow(train_image)

test_image = cv2.imread("./TestWithoutRotations/images/test_image_1.png")

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111)
ax.imshow(test_image)

orb = cv2.SIFT_create()

kp1, des1 = orb.detectAndCompute(train_image, None)
kp2, des2 = orb.detectAndCompute(test_image, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(train_image, kp1, test_image, kp2, good[:100], None, flags=2)

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111)
ax.imshow(img3)
plt.show()
