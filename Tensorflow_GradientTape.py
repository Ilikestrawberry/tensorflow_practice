import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

t1 = tf.constant([1, 2, 3], dtype=tf.float32)  ## input은 constant
t2 = tf.Variable([10, 20, 30], dtype=tf.float32)

with tf.GradientTape() as tape:
    t3 = t1 * t2

gradients = tape.gradient(t3, [t1, t2])
print(gradients)
print(type(gradients))
print('dt1:' ,gradients[0])
print('dt2:' ,gradients[1])

## constant vector는 gradient가 안됨.

# %%

x_data = tf.random.normal(shape=(1000, ), dtype=tf.float32)
y_data = 3*x_data + 1

w = tf.Variable(-1.)  ## float
b = tf.Variable(-1.)

learning_rate = 0.01

w_trace, b_trace = [], []
EPOCHS = 10
for epoch in range(EPOCHS):  ## 전체 학습횟수
    for x, y in zip(x_data, y_data):  ## 위에서 생성한 x,y data 가져옴
        with tf.GradientTape() as tape:  ## forward propagation을 전부 tape에 기록
            prediction = w*x + b  ## Model
            loss = (prediction - y)**2  ## Model과 input값의 차이(Loss)

        gradients = tape.gradient(loss, [w, b])  ## dw/dl, db/dl

        w_trace.append(w.numpy())
        b_trace.append(b.numpy())
        w = tf.Variable(w - learning_rate*gradients[0])  ## w값 업데이트
        b = tf.Variable(b - learning_rate*gradients[1])

# %%

fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(w_trace, label='weight')
ax.plot(b_trace, label='bias')
ax.tick_params(labelsize=20)
ax.legend(fontsize=10)
ax.grid()
plt.show()