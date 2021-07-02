import tensorflow as tf
import numpy as np

## toy dataset 만들기
# train_x = np.arange(1000).astype((np.float32)).reshape(-1, 1)
# train_y = 3*train_x + 1
#
# train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))  ## list, numpyarray -> dataset
# train_ds = train_ds.shuffle(100).batch(32)
#
# for x, y in train_ds:
#     print(x.shape, y.shape, '\n')

# %%
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.data import Dataset

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print(type(train_images))  ## numpyarray type
# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)

train_ds = Dataset.from_tensor_slices((train_images, train_labels))  ## dataset으로 변환
train_ds = train_ds.shuffle(60000).batch(9)  ## 3 X 3 이미지로 띄울거라서 9개씩

test_ds = Dataset.from_tensor_slices((test_images, test_labels))
test_ds = train_ds.batch(32)

train_ds_iter = iter(train_ds)  ## 나열
images, labels = next(train_ds_iter)  ## 하나씩

# print(images.shape)
# print(labels.shape)

fig, axes = plt.subplots(3, 3, figsize=(7, 7))  ## 하나의 axes에는 9개의 이미지가 들어있음


for ax_idx, ax in enumerate(axes.flat):
    image = images[ax_idx, ...]  ## 9개 이미지 중에서 하나씩
    label = labels[ax_idx]
    # print(type(label))

    ax.imshow(image.numpy(), 'gray')
    ax.set_title(label.numpy(), fontsize=10)

    ax.get_xaxis().set_visible(False)  ## 눈금선 없앰
    ax.get_yaxis().set_visible(False)
    plt.show()