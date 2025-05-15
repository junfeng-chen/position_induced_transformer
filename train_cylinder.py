from pit import *
from timeit import default_timer
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.io import loadmat, savemat
from utils import *

def load_data(file_path_train, file_path_test, ntrain, ntest):
    data_train  = loadmat(file_path_train)['trajectories'].astype('float32')# (1000, 4390, 3, 11) 
    data_test   = loadmat(file_path_test)['trajectories'].astype('float32')# (100, 4390, 3, 11)
    trainX = data_train[:ntrain,:,:,:-1].transpose(0,3,1,2).reshape(-1, 4390, 3) # ntrain 1000
    trainY = data_train[:ntrain,:,:,1:].transpose(0,3,1,2).reshape(-1, 4390, 3)
    testX = data_test[:ntest,:,:,:-1].transpose(0,3,1,2).reshape(-1, 4390, 3)
    testY = data_test[:ntest,:,:,1:].transpose(0,3,1,2).reshape(-1, 4390, 3)

    return torch.from_numpy(trainX), torch.from_numpy(trainY), torch.from_numpy(testX), torch.from_numpy(testY)

class pit_cylinder(pit_fixed):
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
        super(pit_cylinder, self).__init__(space_dim,  
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
        mesh_out: (L, self.space_dim)
        '''
        x        = func_in
        size     = mesh_out.shape[:-1]
        mesh_in  = mesh_in.reshape(-1, self.space_dim)
        mesh_out = mesh_out.reshape(-1, self.space_dim)

        func_in  = torch.cat((torch.tile(mesh_in.unsqueeze(0), [func_in.shape[0],1,1]), func_in),-1)
        func_ltt = self.encoder(mesh_in, func_in, self.mesh_ltt)
        func_ltt = self.processor(func_ltt, self.mesh_ltt)
        func_out = self.decoder(self.mesh_ltt, func_ltt, mesh_out)
        return func_out + x

################################################################
ntrain = 1000
ntest = 100
batch_size = 200
learning_rate = 0.001
epochs = 500
iterations = epochs*(ntrain*10//batch_size)
################################################################
# load data and data normalization
################################################################
x_train, y_train, x_test, y_test = load_data("./WakeCylinder_train.mat", "./WakeCylinder_test.mat", ntrain, ntest)
mesh     = torch.from_numpy(np.genfromtxt("vertices.csv", delimiter=",").astype("float32")).to('cuda')# (4390, 2) 
mesh_ltt = torch.from_numpy(np.genfromtxt("vertices_small.csv", delimiter=",").astype("float32")).to('cuda')# (896, 2) 
elements = np.genfromtxt("elements.csv", delimiter=",").astype("int32")-1# (8552, 3)
steps    = y_train.shape[-1]
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, mesh.shape, mesh_ltt.shape, elements.shape)
######
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
################################################################
# training and evaluation
################################################################
model = pit_cylinder(space_dim=2,
                     in_dim=3,
                     out_dim=3,
                     hid_dim=256,
                     n_head=1,
                     n_blocks=4,
                     mesh_ltt=mesh_ltt,
                     en_loc=0.01,
                     de_loc=0.01).cuda()
model = torch.compile(model)
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
  
myloss = RelLpNorm(out_dim=3, p=2)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        loss = 0.
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(mesh, x, mesh)
        loss += myloss(out, y)
        # for t in range(steps):
        #     out = model(mesh, x, mesh)
        #     loss += myloss(out, y[..., t])
        #     x = out
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
            out = model(mesh, x, mesh)
            loss += myloss(out, y)
            # for t in range(steps):
            #     out = model(mesh, x, mesh)
            #     loss += myloss(out, y[..., t])
            #     x = out
            test_l2 += loss.item()

    # train_l2/= (ntrain*steps)
    # test_l2 /= (ntest*steps)
    train_l2/= (10*ntrain)
    test_l2 /= (10*ntest)
    t2 = default_timer()
    print(ep, t2-t1, train_l2, test_l2)
torch.save({'model_state': model.state_dict()}, 'model.pth')
# checkpoint = torch.load('model.pth', weights_only=True)
# model = torch.compile(model)
# model.load_state_dict(checkpoint['model_state'])
# model = torch._dynamo.disable(model)
############################## do evaluation
y_test   = loadmat("./WakeCylinder_test.mat")['trajectories'].astype('float32')
y_test   = torch.from_numpy(y_test)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_test[...,0], y_test[...,1:]), batch_size=100, shuffle=False)
steps    = y_test.shape[-1] - 1
pred = torch.zeros_like(y_test).cuda()
pred[...,0] = y_test[...,0]
i = 0
with torch.no_grad():
    for x, y in test_loader:
        loss = 0.
        x, y = x.cuda(), y.cuda()
        print(i)
        for t in range(steps):
            out = model(mesh, pred[batch_size*i:batch_size*(i+1),:,:,t], mesh)
            loss += myloss(out, y[...,t])
            pred[batch_size*i:batch_size*(i+1),:,:,t+1] = out
        i += 1
        
savemat('pred.mat', mdict={'true':y_test.detach().cpu().numpy(), 'pred':pred.detach().cpu().numpy()})
rel_err = loss / (steps*y_test.shape[0])
print(rel_err)

######### plots
index   = 89
y_test  = y_test.detach().cpu().numpy()
pred    = pred.detach().cpu().numpy()
err     = y_test - pred
abs_err = abs(err[index,...])
emax    = abs_err.max(axis=(0,2))
emin    = abs_err.min(axis=(0,2))
print("error range", emax, emin)
y_test   = y_test[index,...]
vmax    = y_test.max(axis=(0,2))
vmin    = y_test.min(axis=(0,2))
print("vorticity range", vmax, vmin)
pred = pred[index,...]

save_path="."
triangulation = tri.Triangulation(mesh[:,0].cpu(), mesh[:,1].cpu(), elements)
for d in range(3):
    print("Plot variable {}.".format(d+1))
    for t in range(steps+1):
        print("    Plot time {}.".format(t))
        plt.figure(figsize=(8,4),dpi=100)
        plt.axes([0,0,1,1])
        plt.tricontourf(triangulation, y_test[:,d,t], vmax=vmax[d], vmin=vmin[d], levels=512, cmap='plasma')
        plt.axis('off')
        plt.axis('equal')
        plt.savefig(save_path+'/true_variable{}_time{}.pdf'.format(d+1,t))
        plt.close()

        plt.figure(figsize=(8,4),dpi=100)
        plt.axes([0,0,1,1])
        plt.tricontourf(triangulation, pred[:,d,t], vmax=vmax[d], vmin=vmin[d], levels=512, cmap='plasma')
        plt.axis('off')
        plt.axis('equal')
        plt.savefig(save_path+'/pred_variable{}_time{}.pdf'.format(d+1,t))
        plt.close()

        plt.figure(figsize=(8,4),dpi=100)
        plt.axes([0,0,1,1])
        plt.tricontourf(triangulation, abs_err[:,d,t], vmax=emax[d], vmin=emin[d], levels=512, cmap='plasma')
        plt.axis('off')
        plt.axis('equal')
        plt.savefig(save_path+'/err_variable{}_time{}.pdf'.format(d+1,t))
        plt.close()