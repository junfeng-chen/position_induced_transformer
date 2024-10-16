from pit import *
from timeit import default_timer
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from utils import *

def load_data(path_data, ntrain = 1024, ntest=128):

    data          = loadmat(path_data)

    X_data        = data["x"].astype('float32')
    X_data[...,2] = (X_data[...,2]-0.5*X_data[...,1]**2/X_data[...,0])*(1.4-1) # the primitive variable: pressure
    X_data[...,1] = X_data[...,1]/X_data[...,0] # the primitive variable: velocity
    Y_data        = data["y"].astype('float32')
    Y_data[...,2] = (Y_data[...,2]-0.5*Y_data[...,1]**2/Y_data[...,0])*(1.4-1)
    Y_data[...,1] = Y_data[...,1]/Y_data[...,0]
    X_train = X_data[:ntrain,:]
    Y_train = Y_data[:ntrain,:]
    X_test  = X_data[-ntest:,:]
    Y_test  = Y_data[-ntest:,:]
    return torch.from_numpy(X_train), torch.from_numpy(Y_train), torch.from_numpy(X_test), torch.from_numpy(Y_test)

class pit_sod(pit_fixed):
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
        super(pit_sod, self).__init__(space_dim,  
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
        ext: (batch_size, L, self.space_dim)
        '''
        func_in  = torch.cat((torch.tile(mesh_in.unsqueeze(0), [func_in.shape[0],1,1]), func_in),-1)
        func_ltt = self.encoder(mesh_in, func_in, self.mesh_ltt)
        func_ltt = self.processor(func_ltt, self.mesh_ltt)
        func_out = self.decoder(self.mesh_ltt, func_ltt, mesh_out)
        return func_out

ntrain        = 1024
ntest         = 128
batch_size    = 8
learning_rate = 0.001
epochs        = 500
iterations    = epochs*(ntrain//batch_size)

x_train, y_train, x_test, y_test = load_data('./supplementary_data/data_sod.mat', ntrain, ntest)
mesh = torch.linspace(-5,5,x_train.shape[1]+1)[:-1].reshape(-1,1).cuda()
mesh_ltt = torch.linspace(-5,5,256+1)[:-1].reshape(-1,1).cuda()
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

model = pit_sod(space_dim=1,  
                in_dim=3, 
                out_dim=3, 
                hid_dim=32,
                n_head=1, 
                n_blocks=2, 
                mesh_ltt=mesh_ltt, 
                en_loc=0.02, 
                de_loc=0.02).cuda()
model = torch.compile(model)
print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = RelLpNorm(out_dim=3, p=1)
l2err = RelLpNorm(out_dim=3, p=2)
maxerr = RelMaxNorm(out_dim=3)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(mesh, x, mesh)
        loss = myloss(y, out)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

    model.eval()
    test_loss = 0.0
    test_l2   = 0.0
    test_max  = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(mesh, x, mesh)
            test_loss += myloss(y, out).item()
            # test_l1   += l1err(y, out).item()
            test_l2   += l2err(y,out).item()
            test_max  += maxerr(y,out).item()

    train_loss /= ntrain
    test_loss  /= ntest
    # test_l1    /= ntest
    test_l2    /= ntest
    test_max   /= ntest

    t2 = default_timer()
    print(ep, t2-t1, train_loss, test_loss, test_l2, test_max)

torch.save({'model_state': model.state_dict()}, 'model.pth')
###################################
model.eval()
pred      = np.zeros_like(y_test.numpy())
count     = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        out = model(mesh, x, mesh)
        pred[count*batch_size:(count+1)*batch_size,...] = out.detach().cpu().numpy()
        count += 1

y_test = y_test.numpy()
print("relative l1 error", (np.linalg.norm(y_test-pred, axis=1, ord=1) / np.linalg.norm(y_test, axis=1, ord=1)).mean())
print("relative l2 error", (np.linalg.norm(y_test-pred, axis=1, ord=2) / np.linalg.norm(y_test, axis=1, ord=2)).mean())
print("relative l_inf error", (abs(y_test-pred).max(axis=1) / abs(y_test).max(axis=1)).mean() )
savemat("pred.mat", mdict={'pred':pred, 'trueX':x_test, 'trueY':y_test})


index = -1
true  = y_test[index,...].reshape(-1,3)
pred  = pred[index,...].reshape(-1,3)
mesh  = mesh.detach().cpu().numpy().reshape(-1,)
for i in range(3):
    plt.figure(figsize=(12,12),dpi=100)
    plt.plot(mesh, true[:,i], label='true')
    plt.plot(mesh, pred[:,i], label='pred')
    plt.savefig("{}_pred_{}.pdf".format(index, i+1))
    plt.close()