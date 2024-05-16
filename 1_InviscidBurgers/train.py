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
batch_size   = 5
encode_dim   = 64
out_dim      = 1
n_head       = 2
n_train      = 950
n_test       = 128
qry_res      = 1024
ltt_res      = 1024
en_loc       = 1 # locality paramerter in Encoder
de_local     = 8 # locality paramerter in Encoder
save_path    = './model/'


### load dataset
trainX, trainY, testX, testY= load_data("./1_InviscidBurgers.mat", n_train, n_test)
print(trainX.shape, trainY.shape, testX.shape, testY.shape)

### define a model
m_query   = pairwise_dist(qry_res, qry_res) # pairwise distance matrix for Encoder and Decoder
m_cross   = pairwise_dist(qry_res, ltt_res) # pairwise distance matrix for Encoder and Decoder
m_latent  = pairwise_dist(ltt_res, ltt_res) # pairwise distance matrix for Processor
network   = PiT(m_query, m_cross, m_latent, out_dim, encode_dim, n_head, 1, 8)
#network   = LiteTransformer(m_query, m_cross, out_dim, encode_dim, n_head, 1, 8)
#network   = Transformer(qry_res, out_dim, encode_dim, n_head)

inputs    = tf.keras.Input(shape=(qry_res,trainX.shape[-1]))
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
                          batch_size, n_epochs, verbose=1, # verbose to 0 when nohup, to 1 when run in shell
                          validation_data=(testX, testY),
                          validation_batch_size=128)
end   = time()
print(' ')
print('Training cost is {} seconds per epoch !'.format((end-start)/n_epochs))
print(' ')

### save model
model.save_weights(save_path + 'checkpoints/my_checkpoint')

### plot training history
loss_hist = train_history.history
train_loss = tf.reshape(tf.convert_to_tensor(loss_hist["loss"]), (-1,1))
test_loss = tf.reshape(tf.convert_to_tensor(loss_hist["val_loss"]), (-1,1))
savetxt(save_path + 'training_history.csv', tf.concat([train_loss, test_loss], axis=-1).numpy(), delimiter=',', header='train,test', fmt='%1.16f', comments='')

plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
plt.legend()
plt.yscale('log', base=10)
plt.savefig(save_path + 'training_history.png')
plt.close()

### do some tests
pred = model.predict(testX)
savemat(save_path + 'pred.mat', mdict={'pred':pred, 'X':testX, 'Y':testY})
print(rel_l1_median(testY, pred))
