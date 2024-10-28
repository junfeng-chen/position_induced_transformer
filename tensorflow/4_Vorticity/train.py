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
n_epochs    = 500
lr          = 0.001
batch_size  = 8
encode_dim  = 256
out_dim     = 1
n_head      = 1
n_train     = 1000
n_test      = 200
steps       = 20
en_loc      = 1
de_loc      = 8
# load dataset
trainX, trainY, testX, testY= load_data("./NavierStokes_V1e-4_N1200_T30.mat", n_train, n_test)
print(trainX.shape, trainY.shape, testX.shape, testY.shape)

# define a model
m_cross   = pairwise_dist(64, 64, 16, 16) # pairwise distance matrix for encoder and decoder
m_latent  = pairwise_dist(16, 16, 16, 16) # pairwise distance matrix for processor
network   = PiT(m_cross, m_latent, out_dim, encode_dim, n_head, en_loc, de_loc)
r_network = reccurent_PiT(network, steps)
inputs    = tf.keras.Input(shape=(64,64,trainX.shape[-1]))
outputs   = r_network(inputs)
model     = tf.keras.Model(inputs, outputs)
r_network.summary()
### compile model 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.CosineDecay(lr, n_epochs*(n_train//batch_size))),
                                                 loss=rel_norm_step(steps),
                                                 jit_compile=True)
                                                 
### fit model
start = time()
train_history = model.fit(trainX, trainY, 
                        batch_size, n_epochs, verbose=1,
                        validation_data=(testX, testY),
                        validation_batch_size=20)
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

### evaluation, visualization
from scipy.io import savemat
pred = model.predict(testX)
print(rel_norm_traj(testY,pred))
savemat("pred.mat", mdict={"pred":pred, "true":testY})
