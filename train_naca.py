from pit import *
from timeit import default_timer
import matplotlib.pyplot as plt
from scipy.io import savemat
from utils import *

def load_data(path, ntrain, ntest):
    coords     = np.load(path + "shape_coords.npy").astype("float32")#(N,120,2)
  
    vertices_x = np.load(path + "NACA_Cylinder_X.npy")[...,np.newaxis]
    vertices_y = np.load(path + "NACA_Cylinder_Y.npy")[...,np.newaxis]
    X          = np.concatenate((vertices_x, vertices_y), -1).astype("float32")
    Y          = np.load(path + "NACA_Cylinder_Q.npy")[:,:4,...].transpose(0,2,3,1).astype("float32")
 
    return torch.from_numpy(coords[:ntrain,...]), torch.from_numpy(X[:ntrain,...]), torch.from_numpy(Y[:ntrain,...]), torch.from_numpy(coords[-ntest:,...]), torch.from_numpy(X[-ntest:,...]), torch.from_numpy(Y[-ntest:,...])

class pit_naca(pit):
    def __init__(self,
                 space_dim,  
                 in_dim, 
                 out_dim, 
                 hid_dim,
                 n_head, 
                 n_blocks,
                 mesh_ltt,
                 x_downsample,
                 y_downsample,
                 en_loc, 
                 de_loc):
        super(pit_naca, self).__init__(space_dim,  
                                  in_dim, 
                                  out_dim, 
                                  hid_dim,
                                  n_head, 
                                  n_blocks,
                                  mesh_ltt,
                                  en_loc, 
                                  de_loc)

        self.x_down   = x_downsample
        self.y_down   = y_downsample
        self.x_res    = int(220/x_downsample) + 1
        self.y_res    = int(50/y_downsample) + 1

        self.en_layer = kaiming_mlp(self.n_head * self.in_dim, self.hid_dim, self.hid_dim)

    def forward(self, mesh_in, func_in, mesh_out):
        '''
        func_in: (batch_size, L, self.space_dim)
        ext: (batch_size, h, w, self.space_dim)
        '''
        size      = mesh_out.shape[:-1]
        mesh_ltt, mesh_out = self.ltt_mesh(mesh_out) # (batch_size, l_latent, self.space_dim), (batch_size, l_out, self.space_dim)
        ## Encoder
        func_ltt  = self.encoder(mesh_in, func_in, mesh_ltt)
        # Processor
        func_ltt   = self.processor(func_ltt, mesh_ltt)
        # Decoder
        func_out   = self.decoder(mesh_ltt, func_ltt, mesh_out)
        return func_out.reshape(*size, self.out_dim)

    def ltt_mesh(self, mesh_out):
        mesh_ltt    = mesh_out[:, ::self.x_down, ::self.y_down, :][:, :self.x_res, :self.y_res, :].reshape(mesh_out.shape[0], -1, self.space_dim)
        mesh_out    = mesh_out.reshape(mesh_out.shape[0], -1, self.space_dim)
        return mesh_ltt, mesh_out


ntrain       = 1000
ntest        = 200
batch_size    = 20
learning_rate = 0.001
epochs        = 500
iterations    = epochs*(ntrain//batch_size)

x_train, ext_train, y_train, x_test, ext_test, y_test = load_data('./', ntrain, ntest)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, ext_train, y_train), batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, ext_test, y_test), batch_size=batch_size, shuffle=False)

model = pit_naca(space_dim=2,  
            in_dim=2, 
            out_dim=4, 
            hid_dim=128,
            n_head=1, 
            n_blocks=4,
            mesh_ltt=None,
            x_downsample=4,
            y_downsample=4,
            en_loc=0.02, 
            de_loc=0.02).cuda()
model = torch.compile(model)
print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = RelLpNorm(out_dim=4, p=2)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, ext, y in train_loader:
        x, ext, y = x.cuda(), ext.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x, x, ext)
        loss = myloss(y, out)
        loss.backward()

        optimizer.step()
        scheduler.step()
        train_l2 += loss.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, ext, y in test_loader:
            x, ext, y = x.cuda(), ext.cuda(), y.cuda()
            out = model(x, x, ext)
            test_l2 += myloss(y, out).item()

    train_l2/= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2-t1, train_l2, test_l2)
torch.save({'model_state': model.state_dict()}, 'model.pth')
################################################

# checkpoint = torch.load('model_and_optimizer.pth', weights_only=True)
# model.load_state_dict(checkpoint['model_state'])
######### 
rel1err   = RelLpNorm(out_dim=4, p=1)
rel2err   = RelLpNorm(out_dim=4, p=2)
relMaxerr = RelMaxNorm(out_dim=4)
pred      = torch.zeros_like(y_test, device='cpu')
count     = 0

model.eval()
with torch.no_grad():
    for x, ext, y in test_loader:
        x, ext, y = x.cuda(), ext.cuda(), y.cuda()

        out = model(x, x, ext)
        pred[count*batch_size:(count+1)*batch_size,...] = out.detach().cpu()
        count += 1
print("relative l1 error", rel1err(y_test, pred) / ntest)
print("relative l2 error", rel2err(y_test, pred) / ntest)
print("relative l_inf error", relMaxerr(y_test, pred) / ntest)
savemat("pred.mat", mdict={'pred':pred.numpy(), 'trueX':x_test.numpy(), 'ext':ext_test.numpy(), 'trueY':y_test.numpy()})
##############################
index = -1
nvariables = 4
true = y_test.numpy()[index,40:-40,:20,:].reshape(-1,nvariables)
pred = pred.numpy()[index,40:-40,:20,:].reshape(-1,nvariables) #
err  = abs(true-pred)
emax = err.max(axis=0)
emin = err.min(axis=0)
vmax = true.max(axis=0)
vmin = true.min(axis=0)
print(vmax, vmin, emax, emin)

x = ext_test.numpy()[index,40:-40,:20,0].reshape(-1,1)
y = ext_test.numpy()[index,40:-40,:20,1].reshape(-1,1)
print(x.max(), x.min(), y.max(), y.min())

for i in range(nvariables):
    plt.figure(figsize=(12,12),dpi=100)
    plt.scatter(x, y, c=pred[:,i], cmap="plasma", s=160)
    plt.ylim(-0.5,0.5)
    plt.tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
    plt.tight_layout(pad=0)
    plt.savefig("{}_pred_{}.pdf".format(index, i+1))
    plt.close()

    plt.figure(figsize=(12,12),dpi=100)
    plt.scatter(x, y, c=true[:,i], cmap="plasma", s=160)
    plt.ylim(-0.5,0.5)
    plt.tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
    plt.tight_layout(pad=0)
    plt.savefig("{}_true_{}.pdf".format(index, i+1))
    plt.close()

    plt.figure(figsize=(12,12),dpi=100)
    plt.scatter(x, y, c=err[:,i], cmap="plasma", s=160)
    plt.ylim(-0.5,0.5)
    plt.tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
    plt.tight_layout(pad=0)
    plt.savefig("{}_error_{}.pdf".format(index, i+1))
    plt.close()
