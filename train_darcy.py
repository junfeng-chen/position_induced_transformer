from pit import *
from timeit import default_timer
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
# from utilities3 import *
from utils import *

def load_data(train_path, test_path, downsampling, ntrain, ntest):

    s = int(((421 - 1)/downsampling) + 1)

    train  = loadmat(train_path)
    a  = train["coeff"].astype('float32')
    u  = train["sol"].astype('float32')
    
    trainX = a[:ntrain,::downsampling,::downsampling][:,:s,:s]
    trainY = u[:ntrain,::downsampling,::downsampling][:,:s,:s]

    test  = loadmat(test_path)
    a  = test["coeff"].astype('float32')
    u  = test["sol"].astype('float32')
    testX = a[:ntest,::downsampling,::downsampling][:,:s,:s]
    testY = u[:ntest,::downsampling,::downsampling][:,:s,:s]
    return torch.from_numpy(trainX[...,np.newaxis]), torch.from_numpy(trainY[...,np.newaxis]), torch.from_numpy(testX[...,np.newaxis]), torch.from_numpy(testY[...,np.newaxis])
    
class pit_darcy(pit_fixed):
    def __init__(self,
                 space_dim,  
                 in_dim, 
                 out_dim, 
                 hid_dim,
                 n_head, 
                 n_blocks,
                 mesh_ltt,
                 en_loc, 
                 de_loc):
        super(pit_darcy, self).__init__(space_dim,  
                                  in_dim, 
                                  out_dim, 
                                  hid_dim,
                                  n_head, 
                                  n_blocks,
                                  mesh_ltt,
                                  en_loc, 
                                  de_loc)

    def forward(self, mesh_in, func_in, mesh_out):
        '''
        func_in: (batch_size, L, self.in_dim)
        ext: (batch_size, h, w, self.space_dim)
        '''
        size     = mesh_out.shape[:-1]
        mesh_in  = mesh_in.reshape(-1, self.space_dim)
        func_in  = func_in.reshape(func_in.shape[0], -1, self.in_dim)
        mesh_out = mesh_out.reshape(-1, self.space_dim)
        ## Encoder
        func_in  = torch.cat((torch.tile(mesh_in.unsqueeze(0), [func_in.shape[0],1,1]), func_in),-1)
        func_ltt = self.encoder(mesh_in, func_in, self.mesh_ltt)
        # Processor
        func_ltt   = self.processor(func_ltt, self.mesh_ltt)
        # Decoder
        func_out   = self.decoder(self.mesh_ltt, func_ltt, mesh_out)
        return func_out.reshape(func_in.shape[0], *size, self.out_dim)

################################################################
TRAIN_PATH = './piececonst_r421_N1024_smooth1.mat'
TEST_PATH = './piececonst_r421_N1024_smooth2.mat'
ntrain = 1024
ntest = 100
batch_size = 8
learning_rate = 0.001
epochs = 30
iterations = epochs*(ntrain//batch_size)
r = 10
s = int(((421 - 1)/r) + 1)
################################################################
# load data and data normalization
################################################################
x_train, y_train, x_test, y_test = load_data(TRAIN_PATH, TEST_PATH, r, ntrain, ntest)
x_normalizer = PixelWiseNormalization(x_train)
x_train = x_normalizer.normalize(x_train)
x_test = x_normalizer.normalize(x_test)
y_normalizer = PixelWiseNormalization(y_train)


### This part of code for mesh generation is adapted from Li et al. (Fourier Neural Operator for Parametric Partial Differential Equations)
mesh = []
mesh.append(np.linspace(0, 1, s))
mesh.append(np.linspace(0, 1, s))
mesh = np.vstack([xx.ravel() for xx in np.meshgrid(*mesh)]).T
mesh = mesh.reshape(s,s,2)
mesh = torch.tensor(mesh, dtype=torch.float).cuda()

s_ltt = 16
mesh_ltt = []
mesh_ltt.append(np.linspace(0, 1, s_ltt))
mesh_ltt.append(np.linspace(0, 1, s_ltt))
mesh_ltt = np.vstack([xx.ravel() for xx in np.meshgrid(*mesh_ltt)]).T
mesh_ltt = mesh_ltt.reshape(s_ltt,s_ltt,2)
mesh_ltt = torch.tensor(mesh_ltt, dtype=torch.float).cuda()
#######
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=10, shuffle=False)
################################################################
# training and evaluation
################################################################
model = pit_darcy(space_dim=2,  
                 in_dim=1, 
                 out_dim=1, 
                 hid_dim=64,
                 n_head=2, 
                 n_blocks=4, 
                 mesh_ltt=mesh_ltt,
                 en_loc=0.02, 
                 de_loc=0.02).cuda()
model = torch.compile(model)
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
  
myloss = RelLpNorm(out_dim=1, p=2) # LpLoss(size_average=False)
y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:

        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(mesh, x, mesh)
        out = y_normalizer.denormalize(out)
        loss = myloss(y, out)
        loss.backward()
        optimizer.step()
        train_l2 += loss.item()
        scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:

            x, y = x.cuda(), y.cuda()
            out = model(mesh, x, mesh)
            out = y_normalizer.denormalize(out)
            test_l2 += myloss(y, out).item()

    train_l2/= ntrain
    test_l2 /= ntest
    t2 = default_timer()
    print(ep, t2-t1, train_l2, test_l2)
torch.save({'model_state': model.state_dict()}, 'model.pth')
############################## do zero-shot super-resolution evaluation
model = torch._dynamo.disable(model)
r = 1
s = 421
x_train, y_train, x_test, y_test = load_data(TRAIN_PATH, TEST_PATH, r, ntrain, ntest)
x_test = x_normalizer.normalize(x_test)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=10, shuffle=False)
del x_train, y_train
######

mesh = []
mesh.append(np.linspace(0, 1, s))
mesh.append(np.linspace(0, 1, s))
mesh = np.vstack([xx.ravel() for xx in np.meshgrid(*mesh)]).T
mesh = mesh.reshape(s,s,2)
mesh = torch.tensor(mesh, dtype=torch.float).cuda()
pred = torch.zeros_like(y_test)
batch_size = 10
i = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        print(i)
        out = model(mesh, x, mesh)
        out = y_normalizer.denormalize(out)
        pred[batch_size*i:batch_size*(i+1),...] = out
        i += 1
zssr_err = myloss(y_test, pred) / y_test.shape[0]
savemat('zssr.mat', mdict={'true':y_test.detach().cpu().numpy(), 'pred':pred.detach().cpu().numpy()})
print(zssr_err)

######### plots
index = 89
a = x_normalizer.denormalize(x_test)[index,:,:,0].detach().cpu().numpy()
u = y_test[index,:,:,0].detach().cpu().numpy()
pred = pred[index,:,:,0].detach().cpu().numpy()

abs_err = abs(u-pred) * 10000
emax = np.max(abs_err)
emin = np.min(abs_err)
print("Maximum and minimum error", emax, emin)
amax = np.max(a)
amin = np.min(a)
umax = np.max(u)
umin = np.min(u)
print(amax, amin, umax, umin)

# plot the contours
plt.figure(figsize=(14,4),dpi=300)
plt.subplot(141)
plt.imshow(a, vmax=12, vmin=3, interpolation='spline16', cmap='plasma')
cbar = plt.colorbar(location="bottom", fraction=0.046, pad=0.04, ticks=[3, 12], format='%1.0f')
cbar.ax.tick_params(labelsize=14)  # Enlarge colorbar ticks
plt.axis('off')
plt.axis("equal")
plt.title('Permeability', fontsize=16, fontweight='bold')

plt.subplot(142)
plt.imshow(u, vmax=umax, vmin=umin, interpolation='spline16', cmap='plasma')
cbar = plt.colorbar(location="bottom", fraction=0.046, pad=0.04, ticks=[umin, umax], format='%1.3f')
cbar.ax.tick_params(labelsize=14)  # Enlarge colorbar ticks
plt.axis('off')
plt.axis("equal")
plt.title('Reference', fontsize=16, fontweight='bold')

plt.subplot(143)
plt.imshow(pred, vmax=umax, vmin=umin, interpolation='spline16', cmap='plasma')
cbar = plt.colorbar(location="bottom", fraction=0.046, pad=0.04, ticks=[umin, umax], format='%1.3f')
cbar.ax.tick_params(labelsize=14)  # Enlarge colorbar ticks
plt.axis('off')
plt.axis("equal")
plt.title('Prediction', fontsize=16, fontweight='bold')

plt.subplot(144)
plt.imshow(abs_err, vmax=emax, vmin=0, interpolation='spline16', cmap='plasma')
cbar = plt.colorbar(location="bottom", fraction=0.046, pad=0.04, ticks=[0, emax], format='%1.3f')
cbar.ax.tick_params(labelsize=14)  # Enlarge colorbar ticks
plt.axis('off')
plt.axis("equal")
plt.title('Absolute error ('+r"$\times 10^{-4}$"+")", fontsize=16, fontweight='bold')

plt.subplots_adjust(left=0.0, right=1.0, wspace=0.005)
plt.savefig('./prediction.pdf')
plt.close()