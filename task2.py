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
import copy
import sklearn

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

    # We modified __init__ 
    def __init__(self, root_dir:str=None, split:str=None, images=None, labels=None, transform=None, binary=False):
        if images is not None and labels is not None:
            self.images, self.labels = images, labels
        elif root_dir is not None and split is not None:
            self.images, self.labels = load_data(root_dir, split)
        else:
            raise Exception("Either a root directory and a split, or an array of images and labels must be provided.")
        self.transform = transform
        self.binary = binary
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, target = self.images[:,:,:,index], int(self.labels[index])
        if self.transform:
            img = self.transform(img)
        if self.binary:
            return img, target
        else:
            return img, target-1 # target -1 assuming that there are no 0s
    
    
class Ellie(pl.LightningModule):
    
    class ConvBloc(nn.Sequential):
        
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
    
    def __init__(self, in_features, hidden_features, n_classes):
        super().__init__()
        self.block1 = self.ConvBloc(3, 16)
        self.block2 = self.ConvBloc(16,32)
        self.block3 = self.ConvBloc(32,64)
        self.layer1 = nn.Linear(65536, hidden_features)
        self.layer2 = nn.Linear(hidden_features, n_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        hb1 = self.dropout(self.block1(x))
        hb2 = self.dropout(self.block2(hb1))
        hb3 = self.dropout(self.block3(hb2))
        h = torch.flatten(hb3, start_dim=1)
        h = self.activation(self.layer1(h))
        out = self.layer2(h)
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=5e-3)
        return optimizer
    
    def compute_step(self,batch):
        imgs, labels = batch
        # imgs = imgs.view(imgs.size(0), -1)
        label_logits = self.forward(imgs)
        max_pred ,label_predictions = torch.max(label_logits, dim=1 )
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

    # Looad training datasets
    train_29_imgs, train_29_labels = load_data('./data','train_29_task2')
    train_01_imgs, train_01_labels = load_data('./data','train_01_task2')
    
    # Remove dimension in train29
    train_29_imgs = train_29_imgs[:,:,:,0,:]
    
    # Concat training set
    train_imgs_concat = np.concatenate((train_29_imgs, train_01_imgs), axis=3)
    train_labels_concat = np.concatenate((train_29_labels, train_01_labels), axis=0)
    
    # Create datasets
    train_concat_data = SVHNDataset(images=train_imgs_concat, labels=train_labels_concat, transform = transforms.ToTensor())
    
    # Create Dataloader
    train_concat_loader = DataLoader(train_concat_data, batch_size=64, shuffle=True)

    # Init
    in_features = 32*32*3
    hidden_features = 64
    n_classes = 10 
    
    # Create model for training concatenated dataset
    model_concat = Ellie(in_features,
                hidden_features,
                n_classes)

    # Train model
    tb_logger = pl_loggers.TensorBoardLogger("./logs_task2/concat/")
    trainer = pl.Trainer(gpus=1,max_epochs=10, logger=tb_logger)
    trainer.fit(model_concat, train_concat_loader)
    
    # Reload data 
    train_01_imgs, train_01_labels = load_data('./data','train_01_task2')
    test_01_imgs, test_01_labels = load_data('./data', 'test_01_task2')
    
    # Binarize
    le = sklearn.preprocessing.LabelEncoder()
    le = le.fit(np.concatenate((test_01_labels, train_01_labels)))
    train_01_labels, test_01_labels = le.transform(train_01_labels), le.transform(test_01_labels)
    
    # Datatest and dataloader
    train01_data = SVHNDataset(images=train_01_imgs, labels=train_01_labels, transform = transforms.ToTensor(), binary=True)
    test_data = SVHNDataset(images=test_01_imgs, labels=test_01_labels, transform = transforms.ToTensor(), binary=True)
    train01_loader = DataLoader(train01_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)   
    
    # Modify last layer of the model 
    model =  copy.deepcopy(model_concat)
    model.layer2 = nn.Sequential(nn.Linear(hidden_features, 2), nn.Sigmoid())
    
    # Fine-tune model
    tb_logger = pl_loggers.TensorBoardLogger("./logs_task2/")
    fine_tuner = pl.Trainer(logger=tb_logger, gpus=1, max_epochs=10)
    fine_tuner.fit(model, train01_loader, test_loader)

    # Prune model 
    parameters_to_prune = (
        (model.block1[0][0], 'weight'),
        (model.block1[0][3], 'weight'),
        (model.block2[0][0], 'weight'),
        (model.block2[0][3], 'weight'),
        (model.block3[0][0], 'weight'),
        (model.block3[0][3], 'weight'),
        (model.layer1, 'weight')
    )
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.9527)
    
    # Evaluate 
    model.eval()
    y_pred_list, targets_list = [], []
    for (imgs, targets) in  test_loader:
        y_pred = model(imgs)
        _,y_pred_label = torch.max(y_pred, dim=1)
        y_pred_list += y_pred_label.detach().numpy().tolist()
        targets_list += targets.detach().numpy().tolist()    
    print("Test accuracy after prunning: {:.3f}".format(accuracy_score(targets, y_pred_label)))

    
if __name__ == "__main__":
    main()