from pickletools import optimize
from time import time
import torchvision
import os 
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import scipy.io as sio
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch.nn as nn
from  torch.utils.data import DataLoader
import torchmetrics
from pytorch_lightning import loggers as pl_loggers
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score
import time
from pl_bolts.models import VAE
import sklearn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from traitlets import TraitError


def calc_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def eval_model(model,loader):
    model.eval()
    y_pred_list, targets_list = [], []
    for (imgs, targets) in  loader:
        y_pred = model(imgs)
        _,y_pred_label = torch.max(y_pred, dim=1)
        y_pred_list += y_pred_label.detach().numpy().tolist()
        targets_list += targets.detach().numpy().tolist()    
    return(accuracy_score(targets, y_pred_label))


def load_data(root_dir, split):
    """load_data Load images from the dataset

    Args:
        root_dir (string): root directory 
        split (string): type of split based on the task  

    Returns:
        tuple: set of images and lables
    """
    
    filename = os.path.join(root_dir,'test_32x32.mat')
    if(split.startswith('train') or split.startswith('unlabelled')):
        filename = os.path.join(root_dir,'train_32x32.mat') 
    elif(split.startswith('test')):
        filename = os.path.join(root_dir,'test_32x32.mat')
    
    # Load matrix
    loaded_mat = sio.loadmat(filename)
    
    # Parse images and normalize
    imgs = (loaded_mat['X']/255).astype(np.float32)
    
    # Parse labels, convert to int and create vector
    labels = loaded_mat['y'].astype(np.int64).squeeze()
    
    
    if(split=='train_29_task2'):
        imgs_idx_01 =  np.logical_or(labels==10,labels==1)
        imgs_idx_29 = np.where(np.logical_not(imgs_idx_01))
        imgs = imgs[:,:,:,imgs_idx_29]
        labels = labels[imgs_idx_29]
    elif(split=='test_01_task2' or split=='train_01_task2'):
        imgs_idx_01 =  np.where(np.logical_or(labels==10,labels==1))[0]
        if(split=='train_01_task2'):
            imgs_idx_01 = imgs_idx_01[0:200]
        else:
            imgs_idx_01 = imgs_idx_01[200::]
        imgs = imgs[:,:,:,imgs_idx_01]
        labels = labels[imgs_idx_01]
    if(split=='test_task3'):
        N = 50
        imgs = imgs[:,:,:,0:N]
        labels = labels[0:N]
    print('Loaded SVHN split: {split}'.format(split=split))
    print('-------------------------------------')
    print('Images Size: ' , imgs.shape[0:-1])
    print('Split Number of Images:', imgs.shape[-1])
    print('Split Labels Array Size:', labels.shape)
    print('Possible Labels: ', np.unique(labels))
    return imgs,labels

class SVHNDataset(Dataset):
    """SVHNDataset SVHN Dataset class to parse images and targets

    Args:
        Dataset (Dataset): None
    """

    def __init__(self, 
                 root_dir, 
                 split, 
                 transform=None):
        self.images, self.labels = load_data(root_dir, split)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, target = self.images[:,:,:,index], int(self.labels[index])
        if self.transform:
            img = self.transform(img)
        # img = img.permute(1,2,0)
        return img, target-1 # target -1 assuming that there are no 0s
    

        
class Abby(pl.LightningModule):
    
    def deconv_block(self, in_channels, out_channels):
        nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=(2,2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
    
    def __init__(self):
        super().__init__()
        self.deconv_block1 = self.deconv_block(64, out_channels=64)
        self.deconv_block1 = self.deconv_block(64, out_channels=64)
        self.deconv_block1 = self.deconv_block(64, out_channels=64)
    
    

class Joel(pl.LightningModule):
    
    class conv_block(nn.Sequential):
        
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=(2,2)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                )
    
    class deconv_block(nn.Sequential):
        
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=(2,2)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                )

    
    def __init__(self, in_channels:int=3, n_classes:int=10, kl_coeff:float=1):
        super().__init__()
        self.in_channels=in_channels
        self.encoder = nn.Sequential(
            self.conv_block(self.in_channels, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512)
        )
        self.enc_out_dim=8192
        self.latent_dim=100
        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_dec = nn.Linear(100, 8192)
        self.decoder = nn.Sequential(
            self.deconv_block(512, 512),
            self.deconv_block(512, 256),
            self.deconv_block(256, 128)
        )
        self.conv_preclass = nn.Conv2d(in_channels=128, out_channels=3, stride=1, kernel_size=1)
        self.loss = nn.MSELoss()
        self.kl_coeff=kl_coeff
            
    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z
            
    def forward(self, x):
        x = self.encoder(x)
        x_flat = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x_flat)
        log_var = self.fc_var(x_flat)
        p, q, z = self.sample(mu, log_var)
        z = self.fc_dec(z)
        z = torch.reshape(z, shape=(x.shape))
        z = self.decoder(z)
        x_hat = torch.sigmoid(self.conv_preclass(z))
        return z, x_hat, p, q
    
    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=5e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
        return optimizer
    
    def step(self, batch):
        x, y = batch
        z, x_hat, p, q = self.forward(x)

        recon_loss = self.loss(x_hat, x)

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff
        
        # Check this weight
        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs
    
    def training_step(self, batch):
        loss, logs = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    # def validation_step(self, batch):
    #     loss, logs = self.step(batch)
    #     self.log_dict({f"val_{k}": v for k, v in logs.items()})
    #     return loss

        

    
def main():
    
    # Load splits
    unlabelled_imgs, unlabelled_labels = load_data('./data','unlabelled_task3')
    test_imgs, test_labels = load_data('./data','test_task3')
    
    # Create dataset 
    unlabelled_data = SVHNDataset(root_dir='./data', split='unlabelled_task3', transform=transforms.ToTensor())
    test_data = SVHNDataset(root_dir='./data', split='test_task3', transform=transforms.ToTensor())
    unlabelled_loader = DataLoader(unlabelled_data, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    # Create model 
    vae = Joel(in_channels=3, n_classes=10, kl_coeff=0.001)
    
    # Train model 
    tb_logger = pl_loggers.TensorBoardLogger('./logs_vae_torch/')
    trainer = pl.Trainer(gpus=1, max_epochs=5, logger=tb_logger)
    trainer.fit(vae, unlabelled_loader)
    
    # Save model 
    torch.save(vae.state_dict(), 'vae_torch.h5')
    
    

if __name__ == "__main__":
    main()