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
    
    
class Ellie(pl.LightningModule):
    
    def __init__(self, in_features, hidden_features, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)

        
class Abby(pl.LightningModule):
    
    def __init__(self, in_features, hidden_features, n_classes): 
        super().__init__()
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

class TLOU(pl.LightningModule):

    def forward(self, x):
        hb1 = self.dropout(self.block1(x))
        hb2 = self.dropout(self.block2(hb1))
        hb3 = self.dropout(self.block3(hb2))
        h = torch.flatten(hb3, start_dim=1)
        h = self.activation(self.layer1(h))
        out = self.layer2(h)
        out = out
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=5e-3)
        return optimizer
    
    def compute_step(self,batch):
        imgs, labels = batch
        # imgs = imgs.view(imgs.size(0), -1)
        label_logits = self.forward(imgs)
        _,label_predictions = torch.max(label_logits, dim=1 )
        return self.loss(label_logits,labels), labels, label_predictions
    
    def training_step(self, train_batch, batch_idx):
        loss, labels, label_predictions = self.compute_step(train_batch)
        self.train_accuracy(label_predictions, labels)
        self.log_dict({"train/loss": loss, 'train/acc' : self.train_accuracy}, 
                  on_step=False, 
                  on_epoch=True, 
                  prog_bar=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        loss, labels, label_predictions = self.compute_step(val_batch)
        self.val_accuracy(label_predictions, labels)
        self.log_dict({"val/loss": loss, 'val/acc' : self.val_accuracy}, 
                  on_step=False, 
                  on_epoch=True, 
                  prog_bar=True)
        return loss

    def test_step(self, val_batch, batch_idx):
        loss, labels, label_predictions = self.compute_step(val_batch)
        self.val_accuracy(label_predictions, labels)
        self.log_dict({"val/loss": loss, 'val/acc' : self.val_accuracy}, 
                  on_step=False, 
                  on_epoch=True, 
                  prog_bar=True)
        return loss


def main():
    
    # Load settings file 
    with open('settings.json', 'r') as inFile:
        settings = json.load(inFile)
        
    # Load splits
    unlabelled_imgs, unlabelled_labels = load_data('./data','unlabelled_task3')
    test_imgs, test_labels = load_data('./data','test_task3')
    
    # Create dataset 
    unlabelled_data = SVHNDataset(root_dir='./data', split='unlabelled_task3', transform=transforms.ToTensor())
    test_data = SVHNDataset(root_dir='./data', split='test_task3', transform=transforms.ToTensor())
    unlabelled_loader = DataLoader(unlabelled_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    # Init
    in_features = 32*32*3
    hidden_features = 64
    n_classes = 10 
    
    # Load VAE
    model = VAE(input_height=32)

    # Train model
    tb_logger = pl_loggers.TensorBoardLogger("./logs_task3/")
    trainer = pl.Trainer(gpus=1,max_epochs=10,logger = tb_logger, auto_lr_find=True, auto_scale_batch_size=True)
    trainer.fit(model, unlabelled_loader)
    
    # Save and eval
    torch.save(model, "./vae_resnet18.h5")
    # start=time.time()
    # model_acc = eval_model(model, test_loader)
    # stop=time.time()
    # print("Elapsed time: {:.3f}".format(stop-start))
    # print("Test accuracy: {:.3f}".format(model_acc))
    
    # # Prune model 
    # parameters_to_prune = (
    #     (model.block1[0][0], 'weight'),
    #     (model.block1[0][3], 'weight'),
    #     (model.block2[0][0], 'weight'),
    #     (model.block2[0][3], 'weight'),
    #     (model.block3[0][0], 'weight'),
    #     (model.block3[0][3], 'weight'),
    #     (model.layer1, 'weight')
    # )
    
    # prune.global_unstructured(
    #     parameters_to_prune,
    #     pruning_method=prune.L1Unstructured,
    #     amount=0.9527)
    
    # torch.save(model.state_dict(), "/tmp/model_pruned.h5")
    # start=time.time()
    # model_acc = eval_model(model, test_loader)
    # stop=time.time()
    # print("Elapsed time after prunning: {:.3f}".format(stop-start))
    # print("Test accuracy after prunning: {:.3f}".format(model_acc))

    
if __name__ == "__main__":
    main()