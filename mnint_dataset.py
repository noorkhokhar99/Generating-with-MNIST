from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# mnist.train, mnist.test, mnist.validation

batch_xs, batch_ys = mnist.train.next_batch(100)
data = batch_xs[0]
label = batch_ys[0]
pixels = data.reshape((28,28))

print(data)
print(label)

plt.imshow(pixels)
plt.show()