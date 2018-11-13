import tensorflow as tf
import numpy as np

# 例子1，拟合y_data的函数，权重和偏置分别趋近0.1和0.3
#  np.random.rand(100)生成100个[0,1]之间的随机数，构成1维数组
#  np.random.rand(2,3)生成2行3列的二维数组
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3  # 权重偏置这些不断更新的值用tf变量存储，
#  tf.random_uniform()的参数意义：(shape,min,max) #
#  偏置初始化为0
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = weights * x_data + biases
# 损失函数。tf.reduce_mean()是取均值。square是平方。
loss = tf.reduce_mean(tf.square(y - y_data))  # 用梯度优化方法最小化损失函数。
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
# tf变量是需要初始化的，而且后边计算时还需要sess.run(init)一下
init = tf.global_variables_initializer()
# Session进行计算
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(weights), sess.run(biases))
