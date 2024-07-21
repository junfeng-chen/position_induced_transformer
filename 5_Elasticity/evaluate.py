import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '1'
import matplotlib.pyplot as plt
from time import time
from utils import *

# params ##################################33
encode_dim  = 512
out_dim     = 1
n_head      = 8
n_train     = 1000
n_test      = 200

# load dataset
trainX, trainY, testX, testY= load_data("./", n_train, n_test)
print(m_train.shape, m_test.shape, trainX.shape, trainY.shape, testX.shape, testY.shape)

# define a model
network   = PiT(out_dim, encode_dim, n_head, 2, 2)
inputs1   = tf.keras.Input(shape=m_train.shape[1:])
inputs2   = tf.keras.Input(shape=trainX.shape[1:])
outputs   = network(inputs1,inputs2)
model     = tf.keras.Model([inputs1,inputs2], outputs)
network.summary()


### load model
model.load_weights('./512/checkpoints/my_checkpoint').expect_partial()

### evaluation, visualization
index = 89
pred  = model.predict(testX)
print(rel_norm()(testY, pred))
pred  = pred[index:index+1,...]
true = testY[index:index+1,...]

err  = abs(true-pred)
emax = err.max()
emin = err.min()
vmax = true.max()
vmin = true.min()
print(vmax, vmin, emax, emin)

plt.figure(figsize=(6,6),dpi=300)
plt.axes([0,0,1,1])
x = testX[index,:,:1]
y = testX[index,:,1:2]
plt.scatter(x, y, c=pred, cmap="jet", s=160, vmin=vmin, vmax=vmax)
plt.axis("off")
plt.axis("equal")
plt.savefig("pred.pdf")
plt.close()

plt.figure(figsize=(6,6),dpi=300)
plt.axes([0,0,1,1])
plt.scatter(x, y, c=true, cmap="jet", s=160, vmin=vmin, vmax=vmax)
plt.axis("off")
plt.axis("equal")
plt.savefig("true.pdf")
plt.close()

plt.figure(figsize=(6,6),dpi=300)
plt.axes([0,0,1,1])
plt.scatter(x, y, c=err, cmap="jet", s=160, vmin=emin, vmax=emax)
plt.axis("off")
plt.axis("equal")
plt.savefig("error.pdf")
plt.close()
