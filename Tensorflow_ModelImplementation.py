import tensorflow as tf
import matplotlib.pyplot as plt

## train set
x_train = tf.random.normal(shape=(10, ), dtype=tf.float32)
y_train = 3*x_train + 1 + 0.2*tf.random.normal(shape=(10, ), dtype=tf.float32)

## test set
x_test = tf.random.normal(shape=(3, ), dtype=tf.float32)
y_test = 3*x_test + 1 + 0.2*tf.random.normal(shape=(3, ), dtype=tf.float32)

# fig, ax = plt.subplots(figsize=(5, 5))
# ax.scatter(x_train.numpy(), y_train.numpy())
# ax.tick_params(labelsize=10)
# ax.grid()
# plt.show()


## Keras(sequential) 방식
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, activation='linear')
])  ## 1개의 layer, 선형방정식

model.compile(loss='mean_squared_error', optimizer='SGD')  ## error와 backpropagation
model.fit(x_train, y_train, epochs=50, verbose=2)  ## Model 학습
model.evaluate(x_test, y_test, verbose=2)  ## Model 평가


# %%

## Model subclassing 방식
from termcolor import colored

class LinearPredictor(tf.keras.Model):  ## tf.keras.Model 상속
    def __init__(self):
        super(LinearPredictor, self).__init__()  ## 상속받은 class의 init 실행

        self.d1 = tf.keras.layers.Dense(units=1,
                                        activation='linear')  ## 1개의 layer 생성

    def call(self, x):  ## init에서 생성한 layer 받는 함수
        x = self.d1(x)
        return x


EPOCHS = 10
LR = 0.01

## Model, loss 계산 방식, optimizer 방식 지정
model = LinearPredictor()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=LR)


for epoch in range(EPOCHS):
    for x, y in zip(x_train, y_train):
        x = tf.reshape(x, (1, 1))
        print(x.shape, y.shape)
        with tf.GradientTape() as tape:  ## forwardpropagation
            predictions = model(x)
            loss = loss_object(y, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)  ## backpropagation
        ## model.trainable_variables는 tape에 기록되어 있는 모든 variable
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  ## update

    print(colored('Epoch: ', 'red', 'on_white'), epoch+1)

    template = 'Train Loss: {}\n'
    print(template.format(loss))