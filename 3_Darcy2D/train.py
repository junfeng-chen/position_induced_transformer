import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '1'

from numpy import savetxt
from scipy.io import savemat
import matplotlib.pyplot as plt
from time import time

### custom imports
from utils import *

# params ##################################33
n_epochs     = 500
lr           = 0.001
batch_size   = 8
encode_dim   = 128
out_dim      = 1
n_head       = 2
n_train      = 1024
n_test       = 100
downsampling = 2
qry_res      = int((421-1)/downsampling+1)
ltt_res      = 32
en_loc       = 2
de_loc       = 5
# load dataset
trainX, trainY, testX, testY= load_data("./piececonst_r421_N1024_smooth1.mat", "./piececonst_r421_N1024_smooth2.mat", downsampling, n_train, n_test)
print(trainX.shape, trainY.shape, testX.shape, testY.shape)

### normalize input data
XNormalizer = PixelWiseNormalization(trainX)
trainX = XNormalizer.normalize(trainX)
testX = XNormalizer.normalize(testX)
YNormalizer = PixelWiseNormalization(trainY)

### define a model
m_cross   = pairwise_dist(qry_res, qry_res, ltt_res, ltt_res) # pairwise distance matrix for encoder and decoder
m_latent  = pairwise_dist(ltt_res, ltt_res, ltt_res, ltt_res) # pairwise distance matrix for processor
network   = PiT(m_cross, m_latent, out_dim, encode_dim, n_head, en_loc, de_loc, YNormalizer)
inputs    = tf.keras.Input(shape=(qry_res,qry_res,trainX.shape[-1]))
outputs   = network(inputs)
model     = tf.keras.Model(inputs, outputs)
network.summary()
### compile model 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.CosineDecay(lr, n_epochs*(n_train//batch_size))), 
                                                 loss=rel_norm(), 
                                                 jit_compile=True)
                                                 
### fit model
start = time()
train_history = model.fit(trainX, trainY, 
                        batch_size, n_epochs, verbose=1,
                        validation_data=(testX, testY),
                        validation_batch_size=50)
end   = time()
print(' ')
print('Training cost is {} seconds per epoch !'.format((end-start)/n_epochs))

print(' ')
### save model
model.save_weights('./checkpoints/my_checkpoint')
### training history
loss_hist = train_history.history
train_loss = tf.reshape(tf.convert_to_tensor(loss_hist["loss"]), (-1,1))
test_loss = tf.reshape(tf.convert_to_tensor(loss_hist["val_loss"]), (-1,1))
savetxt('./training_history.csv', tf.concat([train_loss, test_loss], axis=-1).numpy(), delimiter=',', header='train,test', fmt='%1.16f', comments='')

plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
plt.legend()
plt.yscale('log', base=10)
plt.savefig('training_history.png')
plt.close()
