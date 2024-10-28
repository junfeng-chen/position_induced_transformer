import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '1'
from scipy.io import loadmat
from utils import *#load_data, pairwise_dist, PixelWiseNormalization
from numpy import zeros, mean, savetxt
from numpy.linalg import norm
import matplotlib.pyplot as plt

# params ##################################33

# load dataset
data = loadmat('./pred.mat')
testY = data['true']
pred  = data['pred']
rel_err = rel_norm_traj(testY,pred)
print(rel_err.numpy())
err = testY - pred
T = 20
rel_err = norm(err.reshape(200,-1,T), ord=2, axis=1) / norm(testY.reshape(200,-1,T), ord=2, axis=1)
rel_err = mean(rel_err, axis=0)
plt.figure(figsize=(16,9),dpi=100)
plt.plot(tf.range(11,31), rel_err)
plt.xlabel('t')
plt.savefig('./rel_err.pdf')
savetxt('./rel_err.csv', rel_err)
############## plots
index = 0
abs_err = abs(err[index,...])
emax = abs_err.max()
emin = abs_err.min()
print(emax, emin)
omega   = testY[index,...]
vmax    = omega.max()
vmin    = omega.min()
print(vmax, vmin)
omega_p = pred[index,...]

for i in range(T):
    # plot the contours
    plt.figure(figsize=(4,4),dpi=300)
    plt.axes([0,0,1,1])
    plt.imshow(omega[...,i], vmax=vmax, vmin=vmin, interpolation='spline16', cmap='jet')
    plt.axis('off')
    plt.axis('equal')
    plt.savefig('./testY_{}.pdf'.format(i+1))
    plt.close()
    
    plt.figure(figsize=(4,4),dpi=300)
    plt.axes([0,0,1,1])
    plt.imshow(omega_p[...,i], vmax=vmax, vmin=vmin, interpolation='spline16', cmap='jet')
    plt.axis('off')
    plt.axis('equal')
    plt.savefig('./pred_{}.pdf'.format(i+1))
    plt.close()
    
    plt.figure(figsize=(4,4),dpi=300)
    plt.axes([0,0,1,1])
    plt.imshow(abs_err[...,i], vmax=emax, vmin=emin, interpolation='spline16', cmap='jet')
    plt.axis('off')
    plt.axis('equal')
    plt.savefig('./err_{}.pdf'.format(i+1))
    plt.close()
    

