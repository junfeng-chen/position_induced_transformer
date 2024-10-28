from pit import *
from timeit import default_timer
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from utils import *

def load_data(file_path, ntrain, ntest, memory, steps):
    try:
        data  = loadmat(file_path)
    except:
        import mat73
        data = mat73.loadmat(file_path)
    flow  = data['u'].astype('float32')
    del data
    trainX = flow[:ntrain,:,:,:memory] # ntrain 1000
    trainY = flow[:ntrain,:,:,memory:memory+steps]
    testX = flow[-ntest:,:,:,:memory]
    testY = flow[-ntest:,:,:,memory:memory+steps]

    del flow
    return torch.from_numpy(trainX), torch.from_numpy(trainY), torch.from_numpy(testX), torch.from_numpy(testY)

class pit_vorticity(pit_periodic2d):
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
        super(pit_vorticity, self).__init__(space_dim,  
                                  in_dim, 
                                  out_dim, 
                                  hid_dim,
                                  n_head, 
                                  n_blocks,
                                  mesh_ltt,
                                  en_loc, 
                                  de_loc)
        self.norm = nn.InstanceNorm1d(hid_dim)
    def forward(self, mesh_in, func_in, mesh_out):
        '''
        func_in: (batch_size, L, self.in_dim)
        ext: (batch_size, h, w, self.space_dim)
        '''
        size     = mesh_out.shape[:-1]
        mesh_in  = mesh_in.reshape(-1, self.space_dim)
        func_in  = func_in.reshape(func_in.shape[0], -1, self.in_dim)
        mesh_out = mesh_out.reshape(-1, self.space_dim)

        func_in  = torch.cat((torch.tile(mesh_in.unsqueeze(0), [func_in.shape[0],1,1]), func_in),-1)
        func_ltt = self.encoder(mesh_in, func_in, self.mesh_ltt)
        func_ltt = self.norm(func_ltt.permute(0,2,1)).permute(0,2,1)

        func_ltt = self.processor(func_ltt, self.mesh_ltt)
        func_ltt = self.norm(func_ltt.permute(0,2,1)).permute(0,2,1)

        func_out = self.decoder(self.mesh_ltt, func_ltt, mesh_out)
        return func_out.reshape(func_in.shape[0], *size, self.out_dim)

################################################################
ntrain = 1000
ntest = 200
batch_size = 20
learning_rate = 0.001
epochs = 500
iterations = epochs*(ntrain//batch_size)
memory = 10
steps  = 20
################################################################
# load data and data normalization
################################################################
x_train, y_train, x_test, y_test = load_data("./NavierStokes_V1e-4_N1200_T30.mat", ntrain, ntest, memory, steps)
s = 64
mesh = []
mesh.append(np.linspace(0, 1, s+1)[:-1])
mesh.append(np.linspace(0, 1, s+1)[:-1])
mesh = np.vstack([xx.ravel() for xx in np.meshgrid(*mesh)]).T
mesh = mesh.reshape(s,s,2)
mesh = torch.tensor(mesh, dtype=torch.float).cuda()

s_ltt = 16
mesh_ltt = []
mesh_ltt.append(np.linspace(0, 1, s_ltt+1)[:-1])
mesh_ltt.append(np.linspace(0, 1, s_ltt+1)[:-1])
mesh_ltt = np.vstack([xx.ravel() for xx in np.meshgrid(*mesh_ltt)]).T
mesh_ltt = mesh_ltt.reshape(s_ltt,s_ltt,2)
mesh_ltt = torch.tensor(mesh_ltt, dtype=torch.float).cuda()
#######
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
################################################################
# training and evaluation
################################################################
model = pit_vorticity(space_dim=2,  
                     in_dim=10, 
                     out_dim=1, 
                     hid_dim=256,
                     n_head=2, 
                     n_blocks=4, 
                     mesh_ltt=mesh_ltt,
                     en_loc=0.02, 
                     de_loc=0.02).cuda()
model = torch.compile(model)
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
  
myloss = RelLpNorm(out_dim=1, p=2)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        loss = 0.
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        for t in range(steps):
            out = model(mesh, x, mesh)
            loss += myloss(out, y[..., t:t+1])
            x = torch.cat((x[..., 1:], out), dim=-1)
        loss.backward()
        optimizer.step()
        train_l2 += loss.item()
        scheduler.step()

    model.eval()
    test_l2 = 0.
    with torch.no_grad():
        for x, y in test_loader:
            loss = 0.
            x, y = x.cuda(), y.cuda()
            for t in range(steps):
                out = model(mesh, x, mesh)
                loss += myloss(out, y[..., t:t+1])
                x = torch.cat((x[..., 1:], out), dim=-1)
            test_l2 += loss.item()

    train_l2/= (ntrain*steps)
    test_l2 /= (ntest*steps)
    t2 = default_timer()
    print(ep, t2-t1, train_l2, test_l2)
torch.save({'model_state': model.state_dict()}, 'model.pth')
############################## do evaluation
pred = torch.zeros_like(y_test)
i = 0
with torch.no_grad():
    for x, y in test_loader:
        loss = 0.
        x, y = x.cuda(), y.cuda()
        print(i)
        for t in range(steps):
            out = model(mesh, x, mesh)
            loss += myloss(out, y[..., t:t+1])
            x = torch.cat((x[..., 1:], out), dim=-1)
        pred[batch_size*i:batch_size*(i+1),...] = yy
        i += 1
        
savemat('pred.mat', mdict={'true':y_test.detach().cpu().numpy(), 'pred':pred.detach().cpu().numpy()})
print(rel_err)

######### plots
index   = 89
y_test  = y_test.detach().cpu().numpy()
pred    = pred.detach().cpu().numpy()
err     = y_test - pred
abs_err = abs(err[index,...])
emax    = abs_err.max()
emin    = abs_err.min()
print("error range", emax, emin)
omega   = y_test[index,...]
vmax    = omega.max()
vmin    = omega.min()
print("vorticity range", vmax, vmin)
omega_p = pred[index,...]

directory="."
for i in range(steps):
    # plot the contours
    plt.figure(figsize=(4,4),dpi=300)
    plt.axes([0,0,1,1])
    plt.imshow(omega[...,i], vmax=vmax, vmin=vmin, cmap="plasma", interpolation='spline16')
    plt.axis('off')
    plt.axis('equal')
    plt.savefig(directory + '/reference_{}.pdf'.format(i+1))
    plt.close()
    
    plt.figure(figsize=(4,4),dpi=300)
    plt.axes([0,0,1,1])
    plt.imshow(omega_p[...,i], vmax=vmax, vmin=vmin, cmap="plasma", interpolation='spline16')
    plt.axis('off')
    plt.axis('equal')
    plt.savefig(directory + '/pred_{}.pdf'.format(i+1))
    plt.close()
    
    plt.figure(figsize=(4,4),dpi=300)
    plt.axes([0,0,1,1])
    plt.imshow(abs_err[...,i], vmax=emax, vmin=emin, cmap="plasma", interpolation='spline16')
    plt.axis('off')
    plt.axis('equal')
    plt.savefig(directory + '/err_{}.pdf'.format(i+1))
    plt.close()