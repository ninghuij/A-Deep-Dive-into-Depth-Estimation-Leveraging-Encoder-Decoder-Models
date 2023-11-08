import torch
from torch import nn
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image 
import cv2
import gc 
from pytorch_msssim import ssim

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import csv
import os

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.base_model = models.efficientnet_b5( pretrained = True )

        filter_size1 = 3
        filter_size2 = 3
        num_features = 2048
        num_features0 = num_features//1
        num_features1 = num_features//1 + 304
        num_features2 = num_features//2

        num_features3 = num_features//2 + 176
        num_features4 = num_features//4

        num_features5 = num_features//4 + 64
        num_features6 = num_features//8

        num_features7 = num_features//8 + 40
        num_features8 = num_features//16

        num_features9 = num_features//16 + 24
        num_features10 = num_features//32


        self.leakyrelu = nn.LeakyReLU(0.2)
        self.layer_conv0 = nn.Conv2d(num_features, num_features0, kernel_size=1, stride=1, padding=0)

        self.layer_conv1 = nn.Conv2d(num_features1 , num_features2, (filter_size1, filter_size1), stride=1, padding=1)#, dilation=1  
        self.layer_conv2 = nn.Conv2d(num_features2, num_features2, (filter_size2, filter_size2), stride=1, padding=1)

        self.layer_conv3 = nn.Conv2d(num_features3 , num_features4, (filter_size1, filter_size1), stride=1, padding=1)  
        self.layer_conv4 = nn.Conv2d(num_features4, num_features4, (filter_size2, filter_size2), stride=1, padding=1)

        self.layer_conv5 = nn.Conv2d(num_features5 , num_features6, (filter_size1, filter_size1), stride=1, padding=1)  
        self.layer_conv6 = nn.Conv2d(num_features6, num_features6, (filter_size2, filter_size2), stride=1, padding=1)
        
        self.layer_conv7 = nn.Conv2d(num_features7 , num_features8, (filter_size1, filter_size1), stride=1, padding=1)  
        self.layer_conv8 = nn.Conv2d(num_features8, num_features8, (filter_size2, filter_size2), stride=1, padding=1)

        self.layer_conv9 = nn.Conv2d(num_features9 , num_features10, (filter_size1, filter_size1), stride=1, padding=1)  
        self.layer_conv10 = nn.Conv2d(num_features10, num_features10, (filter_size2, filter_size2), stride=1, padding=1)

        self.layer_conv11 = nn.Conv2d(num_features10, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # encoder
        features = [x]
        for _ , v in self.base_model.features._modules.items(): 
            features.append( v(features[-1]) )
        
        # decoder
        f0, f1, f2, f3, f4, f5 = features[2], features[3],features[4], features[6],features[7], features[9]         
        
        output = self.layer_conv0(F.relu(f5))
        output0 = F.interpolate(output, size=[f4.size(2), f4.size(3)], mode='bilinear', align_corners=True)
        output0 = self.leakyrelu(self.layer_conv1(torch.cat([output0, f4], dim=1)))
        output0 = self.leakyrelu(self.layer_conv2(output0))

        output1 = F.interpolate(output0, size=[f3.size(2), f3.size(3)], mode='bilinear', align_corners=True)
        output1 = self.leakyrelu(self.layer_conv3(torch.cat([output1, f3], dim=1)))
        output1 = self.leakyrelu(self.layer_conv4(output1))
        
        output2 = F.interpolate(output1, size=[f2.size(2), f2.size(3)], mode='bilinear', align_corners=True)
        output2 = self.leakyrelu(self.layer_conv5(torch.cat([output2, f2], dim=1)))
        output2 = self.leakyrelu(self.layer_conv6(output2))

        output3 = F.interpolate(output2, size=[f1.size(2), f1.size(3)], mode='bilinear', align_corners=True)
        output3 = self.leakyrelu(self.layer_conv7(torch.cat([output3, f1], dim=1)))
        output3 = self.leakyrelu(self.layer_conv8(output3))

        output4 = F.interpolate(output3, size=[f0.size(2), f0.size(3)], mode='bilinear', align_corners=True)
        output4 = self.leakyrelu(self.layer_conv9(torch.cat([output4, f0], dim=1)))
        output4 = self.leakyrelu(self.layer_conv10(output4))

        output5 = self.layer_conv11(output4)
        return output5

def train(dataloader, model, loss_fn, optimizer,start_num,size):
    #size = len(dataloader.dataset)
    avg_loss = 0
    accuracy = 0
    model.train()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for batch, (X, y) in enumerate(dataloader):
        
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X.float()) 
        y_norm = 1000 / y  # depth norm

        loss = loss_fn(pred, y)

        ssim_loss = torch.clamp((1 - ssim( pred, y_norm, data_range = 100.0, size_average=True)) * 0.5, 0, 1 )

        loss = 1.0 * ssim_loss + 0.1 * loss
        #accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        if batch % 500 == 0:
            loss, current = loss.item(), batch * len(X) + start_num
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = avg_loss/len(dataloader)

    return avg_loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return 100 * correct, test_loss

class combine_dataset():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
def get_image_names(root,datatype):

    train_path = root+'data/nyu2_'+ datatype + '.csv'
    imgs_path = []
    with open(train_path, 'r') as f:
        csvreader = csv.reader(f) 
        for img_path in csvreader:
            color_name = img_path[0]
            depth_name = img_path[1]
            color_path = root + color_name
            depth_path = root + depth_name
            imgs_path.append([color_path,depth_path])
    return imgs_path
def img_transform(img_paths,trans,trans1,start_num,end_num):
    datax =[] 
    datay =[]
    for num in range(end_num - start_num):
        color_path = img_paths[num + start_num][0]
        depth_path = img_paths[num + start_num][1]

        image = cv2.imread(color_path)
        image = trans(image)
        datax.append(np.array(image))

        depth = cv2.imread(depth_path)
        depth = depth[:,:,0] # choose only one channle
        depth = trans1(depth) * 1000
        depth = torch.clamp(depth, 10, 1000)
        datay.append(np.array(depth))

    datasets = combine_dataset(datax, datay)
    #print("Transform is done!")
    return datasets
def preprocess(root,datatype,num_load, trans,trans1):
    print("Start transform.")
    num = 0
    datax =[] 
    datay =[]
    train_path = root+'data/nyu2_'+ datatype + '.csv'
    with open(train_path, 'r') as f:
        csvreader = csv.reader(f) 

        for img_path in csvreader:
            color_name = img_path[0]
            depth_name = img_path[1]

            color_path = root + color_name
            depth_path = root + depth_name

            image = cv2.imread(color_path)
            depth = cv2.imread(depth_path)
            depth = depth[:,:,0] # choose only one channle
            image = trans(image)
            depth = trans1(depth)
            datax.append(np.array(image))
            datay.append(np.array(depth))
            num = num +1
            if num > num_load :
                break

    datasets = combine_dataset(datax, datay)
    print("Transform is done!")
    return datasets
def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[:,0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.

    cmapper = plt.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:,:3]

    return img.transpose((0,3,1,2))
def LogProgress(model, writer, test_dataloader, iters):

    model.eval()
    image, depth = next(iter(test_dataloader))
    if iters == 0:
        writer.add_image('train.1.image', image.data,dataformats='NCHW')
        writer.add_image('train.2.depth', colorize(depth.data), dataformats='NCHW')

    output = model(image.float().to(device))
    # out_images = output.detach().cpu().numpy()
    # out_images = np.expand_dims(out_images, axis = 0) # add a dim
    # out_images = np.swapaxes(out_images,1,0)
    writer.add_image('train.3.output0',  colorize(output.data),dataformats='NCHW')
    output = 1000/output
    writer.add_image('train.3.output1',  colorize(output.data),dataformats='NCHW')
    writer.add_image('train.4.diff', colorize(torch.abs(output.cpu()-depth).data),dataformats='NCHW')

    del image
    del depth
    del output
if __name__ == "__main__":

    # hyperparameters
    learning_rate = 0.0001
    train_batch_size = 1
    iters = 15  # iterations

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # create the network model
    model = Model().to(device)
    #model.load_state_dict(torch.load('trained_model_params.pkl'))
    # Define loss and optimizer
    cost = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=1e-5)

    trans = transforms.Compose([
                                transforms.ToPILImage(),
                                #transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                #transforms.Normalize([123.675,116.28,103.53], [58.395,57.12,57.375])
                                ])
    trans_depth = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((240, 320)),
                                transforms.ToTensor(),
                                ])
    
    os.makedirs('./trained_models', exist_ok = True)


    testset = preprocess(root='./datasets/nyu_data/',datatype = 'test',num_load = 400, trans = trans,trans1=trans_depth)
    test_dataloader = DataLoader(testset, batch_size = train_batch_size, shuffle=True)
    
    writer = SummaryWriter()
    img_paths = get_image_names(root='./datasets/nyu_data/',datatype = 'train')
    # load images to memory by size to avoid of running out memory
    load_size = 4000  # maximum value is len(img_paths)

    # Get the accuracy and loss based on the iterations
    for i in range(iters): 
        print("iters:" + str(i))
        for num in range(len(img_paths)//load_size):
            start_num = num * load_size
            end_num = min((num + 1) * load_size, len(img_paths))
            trainset = img_transform(img_paths,trans = trans,trans1=trans_depth,start_num = start_num,end_num = end_num)
            train_dataloader = DataLoader(trainset, batch_size = train_batch_size, shuffle=True)

            training_loss = train(train_dataloader, model, cost, optimizer,start_num,len(img_paths))
            print(f"Training Error: Avg loss: {training_loss:>8f} \n")


            seq = i* (len(img_paths)//load_size+1) + num
            writer.add_scalars('Loss/', {'training':training_loss}, seq)
            #writer.add_scalars('Accuracy/', {'training':training_accuracy}, seq)
            LogProgress(model,writer,train_dataloader,seq)

            # release memory
            del trainset
            del train_dataloader
            gc.collect()

            save_path = './trained_models/trained_model_params_iter'+ str(i)+'.pkl' #cover the last one
            torch.save(model.state_dict(),save_path)

    save_path = './trained_models/trained_completed_model_params.pkl'
    torch.save(model.state_dict(),save_path)

    writer.close()