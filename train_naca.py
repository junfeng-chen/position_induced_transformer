import torch
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')
from numpy import load, newaxis, random, concatenate
random.seed(0)
from timeit import default_timer
import matplotlib.pyplot as plt
from scipy.io import savemat
from pit import *
from utils import *

def load_data(path, ntrain, ntest):
  
    vertices_x = load(path + "NACA_Cylinder_X.npy")[...,newaxis]
    vertices_y = load(path + "NACA_Cylinder_Y.npy")[...,newaxis]
    X          = concatenate((vertices_x, vertices_y), -1).astype("float32")
   
    Y          = load(path + "NACA_Cylinder_Q.npy")[:,4:,...].transpose(0,2,3,1)
    Y          = Y.astype("float32")
 
    return torch.from_numpy(X[:ntrain,...]), torch.from_numpy(Y[:ntrain,...]), torch.from_numpy(X[-ntest:,...]), torch.from_numpy(Y[-ntest:,...])


n_train       = 1000
n_test        = 200
batch_size    = 8
learning_rate = 0.001
epochs        = 500
iterations    = epochs*(n_train//batch_size)

x_train, y_train, x_test, y_test = load_data('./', n_train, n_test)
print(x_train.size(), y_train.size(), x_test.size(), y_test.size())
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

model = torch.nn.DataParallel(pit(space_dim=2,  
                 in_dim=2, 
                 out_dim=1, 
                 hid_dim=256,
                 n_head=2,
                 n_blocks=4,
                 l_latent=111*26,
                 en_loc=0.01, 
                 de_loc=0.01).cuda(), device_ids = [0,1,2,3])

print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=iterations, final_div_factor=1000.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = RelLpNorm(out_dim=1, p=2)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

        loss = myloss(y, out)
        loss.backward()

        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            test_loss += myloss(y, out).item()

    train_loss /= n_train
    test_loss  /= n_test

    t2 = default_timer()
    print(ep, t2-t1, train_loss, test_loss)
torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, 'model_and_optimizer.pth')
##############

# checkpoint = torch.load('model_and_optimizer.pth')
# model.load_state_dict(checkpoint['model_state'])
rel1err   = RelLpNorm(out_dim=1, p=1)
rel2err   = RelLpNorm(out_dim=1, p=2)
relMaxerr = RelMaxNorm(out_dim=1)
pred      = torch.zeros_like(y_test, device='cpu')
count     = 0

model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()

        out = model(x)
        pred[count*batch_size:(count+1)*batch_size,...] = out.detach().cpu()
        count += 1
print("relative l1 error", rel1err(y_test, pred) / n_test)
print("relative l2 error", rel2err(y_test, pred) / n_test)
print("relative l_inf error", relMaxerr(y_test, pred) / n_test)
savemat("pred.mat", mdict={'pred':pred.numpy(), 'trueX':x_test.numpy(), 'trueY':y_test.numpy()})

index = -1
true = y_test.numpy()[index,40:-40,:20,:].reshape(-1,1)
pred = pred.numpy()[index,40:-40,:20,:].reshape(-1,1) #

err  = abs(true-pred)
emax = err.max(axis=0)
emin = err.min(axis=0)
vmax = true.max(axis=0)
vmin = true.min(axis=0)
print(vmax, vmin, emax, emin)

x = x_test.numpy()[index,40:-40,:20,0].reshape(-1,1)
y = x_test.numpy()[index,40:-40,:20,1].reshape(-1,1)
print(x.max(), x.min(), y.max(), y.min())

for i in range(1):
    plt.figure(figsize=(12,12),dpi=100)
    plt.scatter(x, y, c=pred[:,i], cmap="jet", s=160)
    plt.ylim(-0.5,0.5)
    plt.tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
    plt.tight_layout(pad=0)
    plt.savefig("{}_pred_{}.pdf".format(index, i))
    plt.close()

    plt.figure(figsize=(12,12),dpi=100)
    plt.scatter(x, y, c=true[:,i], cmap="jet", s=160)
    plt.ylim(-0.5,0.5)
    plt.tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
    plt.tight_layout(pad=0)
    plt.savefig("{}_true_{}.pdf".format(index, i))
    plt.close()

    plt.figure(figsize=(12,12),dpi=100)
    plt.scatter(x, y, c=err[:,i], cmap="jet", s=160)
    plt.ylim(-0.5,0.5)
    plt.tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
    plt.tight_layout(pad=0)
    plt.savefig("{}_error_{}.pdf".format(index, i))
    plt.close()
