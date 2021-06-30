import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

plt.style.use('seaborn')

n_sample = 100
x_train = np.random.normal(0, 1, size=(n_sample, 1)).astype(np.float32)
y_train = (x_train >= 0).astype(np.float32)  ## 0 or 1

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(x_train, y_train)
ax.tick_params(labelsize=10)
plt.show()

# %%

class classifier(tf.keras.Model):
    def __init__(self):
        super(classifier, self).__init__()

        self.d1 = tf.keras.layers.Dense(units=1,
                                        activation='sigmoid')

    def call(self, x):
        predictions = self.d1(x)
        return predictions

EPOCHS = 10
LR = 0.01
model = classifier()
loss_object = tf.keras.losses.BinaryCrossentropy()  ## Binary loss 계산
optimizer = tf.keras.optimizers.SGD(learning_rate=LR)

loss_metric = tf.keras.metrics.Mean()  ## loss의 평균을 계산
acc_metric = tf.keras.metrics.CategoricalAccuracy()  ## accuracy

for epoch in range(EPOCHS):
    for x, y in zip(x_train, y_train):
        x = tf.reshape(x, (1, 1))
        y = tf.reshape(y, (1, 1))

        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_metric(loss)  ## loss는 각 train 단계의 loss값이기 때문에 평균을 계산해서 보는게 더 정확
        acc_metric(y, predictions)  ## accuracy 계산

    print(colored('Epoch: ', 'cyan', 'on_white'), epoch + 1)
    template = 'Train Loss: {:.4f}\t Train Accuracy: {:.2f}%'

    ds_loss = loss_metric.result()  ## 계산값 받음
    ds_acc = acc_metric.result()

    print(template.format(ds_loss,
                          ds_acc*100))

    loss_metric.reset_state()  ## 각 epoch마다 초기화
    acc_metric.reset_state()

# %%

x_min, x_max = x_train.min(), x_train.max()

x_test = np.linspace(x_min, x_max, 300).astype(np.float32).reshape(-1, 1)

x_test_tf = tf.constant(x_test)
y_test_tf = model(x_test_tf)

x_result = x_test_tf.numpy()
y_result = y_test_tf.numpy()

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(x_train, y_train)
ax.tick_params(labelsize=10)
ax.plot(x_result, y_result,
        "r:",
        linewidth=3, )
plt.show()