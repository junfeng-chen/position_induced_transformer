import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '1'

from numpy import savetxt
import matplotlib.pyplot as plt
from time import time

### custom imports
from utils import *

# params ##################################33
encode_dim  = 256
out_dim     = 1
n_head      = 2
n_train     = 1000
n_test      = 200

# load dataset
trainX, trainY, testX, testY= load_data("./", ntrain=n_train, ntest=n_test)
print(trainX.shape, trainY.shape, testX.shape, testY.shape)

# define a model
m_cross   = pairwise_dist(51, 221, 26, 111) # pairwise distance matrix for encoder and decoder
m_latent  = pairwise_dist(26, 111, 26, 111) # pairwise distance matrix for processor
network   = PiT(m_cross, m_latent, out_dim, encode_dim, n_head, 0.5, 2)
inputs    = tf.keras.Input(shape=(221,51,trainX.shape[-1]))
outputs   = network(inputs)
model     = tf.keras.Model(inputs, outputs)
network.summary()
print(' ')
### load model
model.load_weights('./256_0.5_2_selected/checkpoints/my_checkpoint').expect_partial()

### evaluation, visualization
pred = model.predict(testX)

index = -67
true = testY[index,40:-40,:20,0].reshape(-1,1)
pred = model(testX)[index,40:-40,:20,0].numpy().reshape(-1,1)
err  = abs(true-pred)
emax = err.max()
emin = err.min()
vmax = true.max()
vmin = true.min()
print(vmax, vmin, emax, emin)

x = testX[index,40:-40,:20,0].reshape(-1,1)
y = testX[index,40:-40,:20,1].reshape(-1,1)
print(x.max(), x.min(), y.max(), y.min())

plt.figure(figsize=(12,12),dpi=100)
plt.scatter(x, y, c=pred, cmap="jet", s=160)
plt.ylim(-0.5,0.5)
plt.tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
plt.tight_layout(pad=0)
plt.savefig("pred.pdf")
plt.close()

plt.figure(figsize=(12,12),dpi=100)
plt.scatter(x, y, c=true, cmap="jet", s=160)
plt.ylim(-0.5,0.5)
plt.tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
plt.tight_layout(pad=0)
plt.savefig("true.pdf")
plt.close()

plt.figure(figsize=(12,12),dpi=100)
plt.scatter(x, y, c=err, cmap="jet", s=160)
plt.ylim(-0.5,0.5)
plt.tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
plt.tight_layout(pad=0)
plt.savefig("error.pdf")
plt.close()

plt.figure(figsize=(12,12),dpi=100)
plt.scatter(x, y, color="black", s=160)
plt.ylim(-0.5,0.5)
plt.tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
plt.tight_layout(pad=0)
plt.savefig("points.pdf")
plt.close()
