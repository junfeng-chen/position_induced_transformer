import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
from time import time
from utils import *

# params ##################################33
n_epochs    = 500
lr          = 0.001
batch_size  = 10
encode_dim  = 512
out_dim     = 1
n_head      = 8
n_train     = 1000
n_test      = 200

# load dataset
trainX, trainY, testX, testY= load_data("./", n_train, n_test)
print(trainX.shape, trainY.shape, testX.shape, testY.shape)

# define a model
network   = PiT(out_dim, encode_dim, n_head, 2, 2)
inputs   = tf.keras.Input(shape=trainX.shape[1:])
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
np.savetxt('./training_history.csv', tf.concat([train_loss, test_loss], axis=-1), delimiter=',', header='train,test', fmt='%1.16f', comments='')

plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
plt.legend()
plt.yscale('log', base=10)
plt.savefig('training_history.png')
plt.close()

### evaluation, visualization
index = 2
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
