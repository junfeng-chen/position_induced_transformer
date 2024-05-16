import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '1'
from scipy.io import savemat
from utils import *#load_data, pairwise_dist, PixelWiseNormalization
from numpy import zeros
import matplotlib.pyplot as plt
downsampling = 10
qry_res      = int((421-1)/downsampling+1)
ltt_res      = 32
en_loc       = 2
de_loc       = 5
encode_dim   = 128
out_dim      = 1
n_head       = 2
n_train      = 1024
n_test       = 100


## Load training data and construct normalizer
trainX, trainY, testX, testY= load_data("./piececonst_r421_N1024_smooth1.mat", "./piececonst_r421_N1024_smooth2.mat", downsampling, n_train, n_test)
print(trainX.shape, trainY.shape, testX.shape, testY.shape)
XNormalizer = PixelWiseNormalization(trainX)
YNormalizer = PixelWiseNormalization(trainY)

## creat model, load trained weights
m_cross   = pairwise_dist(qry_res, qry_res, ltt_res, ltt_res) # pairwise distance matrix for encoder and decoder
m_latent  = pairwise_dist(ltt_res, ltt_res, ltt_res, ltt_res) # pairwise distance matrix for processor
network   = PiT(m_cross, m_latent, out_dim, encode_dim, n_head, en_loc, de_loc, YNormalizer)
inputs    = tf.keras.Input(shape=(qry_res,qry_res,trainX.shape[-1]))
outputs   = network(inputs)
model     = tf.keras.Model(inputs, outputs)
model.load_weights('./results_43/checkpoints/my_checkpoint').expect_partial()

#testX = XNormalizer.normalize(testX)
#pred = tf.convert_to_tensor(model.predict(testX), "float32")
#rel_err = rel_norm()(testY,pred)
#print(rel_err)
#pred = tf.convert_to_tensor(network(testX), "float32")
#rel_err = rel_norm()(testY,pred)
#print(rel_err)

#network.summary()
##################
#zeor-shot super-resolution
downsampling = 1
qry_res      = int((421-1)/downsampling+1)
trainX, trainY, testX, testY= load_data("./piececonst_r421_N1024_smooth1.mat", "./piececonst_r421_N1024_smooth2.mat", downsampling, n_train, n_test)
m_cross   = pairwise_dist(qry_res, qry_res, ltt_res, ltt_res) # pairwise distance matrix for encoder and decoder
network2    = PiT(m_cross, m_latent, out_dim, encode_dim, n_head, 2, 5, YNormalizer)
inputs2    = tf.keras.Input(shape=(qry_res,qry_res,trainX.shape[-1]))
outputs2   = network2(inputs2)
model2     = tf.keras.Model(inputs2, outputs2)
model2.set_weights(model.get_weights()) 

testX = XNormalizer.normalize(testX)
pred = tf.convert_to_tensor(model2.predict(testX), "float32")
rel_err = rel_norm()(testY,pred)
print(rel_err)

######### plots
index = 43
a = XNormalizer.denormalize(testX)[index,:,:,0]
u = testY[index,:,:,0]
pred = pred[index,:,:,0]

abs_err = tf.math.abs(u-pred) * 10000
emax = tf.math.reduce_max(abs_err)
emin = tf.math.reduce_min(abs_err)
print(emax, emin)
amax = tf.math.reduce_max(a)
amin = tf.math.reduce_min(a)
umax = tf.math.reduce_max(u)
umin = tf.math.reduce_min(u)
print(amax, amin, umax, umin)

# plot the contours
plt.figure(figsize=(17,4),dpi=300)
plt.subplot(141)
plt.imshow(a, vmax=12, vmin=3, interpolation='spline16', cmap='jet')
plt.colorbar(location="bottom", fraction=0.046, pad=0.04, ticks=[3, 12], format='%1.0f')
plt.axis('off')
plt.axis("equal")
plt.title('Permeability')

plt.subplot(142)
plt.imshow(u, vmax=umax, vmin=umin, interpolation='spline16', cmap='jet')
plt.colorbar(location="bottom", fraction=0.046, pad=0.04, ticks=[umin, umax], format='%1.3f')
plt.axis('off')
plt.axis("equal")
plt.title('Reference')

plt.subplot(143)
plt.imshow(pred, vmax=umax, vmin=umin, interpolation='spline16', cmap='jet')
plt.colorbar(location="bottom", fraction=0.046, pad=0.04, ticks=[umin, umax], format='%1.3f')
plt.axis('off')
plt.axis("equal")
plt.title('Prediction')

plt.subplot(144)
plt.imshow(abs_err, vmax=emax, vmin=0, interpolation='spline16', cmap='jet')
plt.colorbar(location="bottom", fraction=0.046, pad=0.04, ticks=[0, emax], format='%1.3f')
plt.axis('off')
plt.axis("equal")
plt.title('Absolute error ('+r"$\times 10^{-4}$"+")")
plt.savefig('./prediction.pdf')
plt.close()


