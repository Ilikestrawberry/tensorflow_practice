import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Mean

n_train, n_validation, n_test = 1000, 300, 300

train_x = np.random.normal(0, 1, size=(n_train, 1)).astype(np.float32)
train_x_noise = train_x + 0.2*np.random.normal(0, 1, size=(n_train, 1))
train_y = (train_x_noise > 0).astype(np.int32)

validation_x = np.random.normal(0, 1, size=(n_validation, 1)).astype(np.float32)
validation_x_noise = validation_x + 0.2*np.random.normal(0, 1, size=(n_validation, 1))
validation_y = (validation_x_noise > 0).astype(np.int32)

test_x = np.random.normal(0, 1, size=(n_test, 1)).astype(np.float32)
test_x_noise = test_x + 0.2*np.random.normal(0, 1, size=(n_test, 1))
test_y = (test_x_noise > 0).astype(np.int32)

# fig, ax = plt.subplots(figsize=(10, 7))
# ax.scatter(train_x, train_y)
# ax.tick_params(labelsize=10)
# ax.grid()
# plt.show()

# %%

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))  ## x,y를 dataset으로
train_ds = train_ds.shuffle(n_train).batch(8)  ## 8개씩 랜덤으로 뽑아서 (8, 1) matrix 형태로 입력

validation_ds = tf.data.Dataset.from_tensor_slices((validation_x, validation_y))
validation_ds = validation_ds.batch(n_validation)  ## 평가는 그냥 한번에 입력

test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_ds = test_ds.batch(n_test)

# %%

model = Sequential()
model.add(Dense(units=2, activation='softmax'))

# class MyModel(Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.d1 = Dense(units=2, activation='softmax')
#
#     def call(self, x):
#         x = self.d1(x)
#         return x

loss_object = SparseCategoricalCrossentropy()
optimizer = SGD(learning_rate=1)

train_loss = Mean()
train_acc = SparseCategoricalAccuracy()

validation_loss = Mean()
validation_acc = SparseCategoricalAccuracy()

test_loss = Mean()
test_acc = SparseCategoricalAccuracy()

EPOCHS = 30

# %%

@tf.function
def train_step(x, y):
    global model, loss_object
    global train_loss, train_acc

    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(y, predictions)

@tf.function
def validation():
    global  validation_ds, model, loss_object
    global validation_loss, validation_acc

    for x, y in validation_ds:
        predictions = model(x)
        loss = loss_object(y, predictions)

        validation_loss(loss)
        validation_acc(y, predictions)

def train_reporter():
    global epoch
    global train_loss, train_acc
    global validation_loss, validation_acc

    print(colored('Epoch: ', 'red', 'on_white'), epoch + 1)
    template = 'Train Loss: {:.4f}\t Train Accuracy: {:.2f}%\n' + \
               'Validation Loss: {:.4f}\t Validation Accuracy: {:.2f}%\n'
    print(template.format(train_loss.result(),
                          train_acc.result() * 100,
                          validation_loss.result(),
                          validation_acc.result() * 100))

def metric_resetter():
    global train_loss, train_acc
    global validation_loss, validation_acc

    train_losses.append(train_loss.result())
    validation_losses.append(validation_loss.result())
    train_accs.append(train_acc.result())
    validation_accs.append(validation_acc.result())

    train_loss.reset_state()
    train_acc.reset_state()
    validation_loss.reset_state()
    validation_acc.reset_state()

def final_result_visualization():
    global train_losses, validation_losses
    global train_accs, validation_accs

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))  ## 가로로 2개 그래프(ax1, ax2)
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(validation_losses, label='Validation Loss')

    axes[1].plot(train_accs, label='Train Accuracy')
    axes[1].plot(validation_accs, label='Validation Accuracy')
    axes[0].tick_params(labelsize=10)
    axes[1].tick_params(labelsize=10)

    axes[0].set_ylabel('Binary Cross Entropy', fontsize=10)
    axes[1].set_ylabel('Accuracy', fontsize=10)
    axes[1].set_xlabel('Epoch', fontsize=10)

    axes[0].legend(loc='upper right', fontsize=10)
    axes[1].legend(loc='upper right', fontsize=10)

    plt.show()

## 이 아래 구조를 잘 숙지해두기.

train_losses, validation_losses = [], []  ## 시각화 할 때 쓸 값들 저장
train_accs, validation_accs = [], []

for epoch in range(EPOCHS):
    for x, y in train_ds:
        train_step(x, y)
        # with tf.GradientTape() as tape:  ## tape에 기록
        #     predictions = model(x)  ## 모델을 이용해서 나온 결과값
        #     loss = loss_object(y, predictions)  ## 결과값과 실제값의 차이
        #
        # gradients = tape.gradient(loss, model.trainable_variables)  ## 편미분
        # optimizer.apply_gradients(zip(gradients, model.trainable_variables))  ## SGD에 적용
        #
        # train_loss(loss)  ## loss의 평균
        # train_acc(y, predictions)

    validation()
    # for x, y in validation_ds:
    #     predictions = model(x)
    #     loss = loss_object(y, predictions)
    #
    #     validation_loss(loss)
    #     validation_acc(y, predictions)

    train_reporter()
    # print(colored('Epoch: ', 'red', 'on_white'), epoch + 1)
    # template = 'Train Loss: {:.4f}\t Train Accuracy: {:.2f}%\n' + \
    #     'Validation Loss: {:.4f}\t Validation Accuracy: {:.2f}%\n'
    # print(template.format(train_loss.result(),
    #                       train_acc.result()*100,
    #                       validation_loss.result(),
    #                       validation_acc.result()*100))

    metric_resetter()
    # train_losses.append(train_loss.result())
    # validation_losses.append(validation_loss.result())
    # train_accs.append(train_acc.result())
    # validation_accs.append(validation_acc.result())
    #
    # train_loss.reset_state()
    # train_acc.reset_state()
    # validation_loss.reset_state()
    # validation_acc.reset_state()

for x, y in test_ds:
    predictions = model(x)
    loss = loss_object(y, predictions)

    test_loss(loss)
    test_acc(y, predictions)

final_result_visualization()
print(colored('Final Result: ', 'cyan', 'on_white'))
template = 'Test Loss: {:.4f}\t Test Accuracy: {:.2f}%\n'
print(template.format(test_loss.result(),
                      test_acc.result()*100))

# %%

# train_loss = Mean()
#
# t1 = tf.constant([1, 2, 3, 4, 5, 6])
# for t in t1:
#     train_loss(t)
#     print(train_loss.result())