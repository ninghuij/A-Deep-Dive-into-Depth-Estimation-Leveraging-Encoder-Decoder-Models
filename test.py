# import matplotlib.pyplot as plt
# import numpy as np

# import torch
# from torch import nn
# from torchvision import datasets
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import torchvision.transforms as transforms
from train import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model_name", type=str, default='trained_model_params_iter1_1.pkl',
        help="pretrained_model_name")
    args = parser.parse_args()
    return args

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

cost = nn.CrossEntropyLoss()
model = Model().to(device)

#load model parameters 
args = parse_args()
input_path = './trained_models/' + args.model_name
#model.load_state_dict(torch.load('./trained_models/trained_completed_model_params.pkl'))
model.load_state_dict(torch.load(input_path))
trans = transforms.Compose([
                            transforms.ToPILImage(),
                            #transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            #transforms.Normalize([123.675,116.28,103.53], [58.395,57.12,57.375])
                            ])
trans_depth = transforms.Compose([
                            transforms.ToPILImage(),
                            #transforms.Resize((240, 320)),
                            transforms.ToTensor(),
                            #transforms.Normalize([123.675,116.28,103.53], [58.395,57.12,57.375])
                            ])

# load data
testset = preprocess(root='./datasets/nyu_data/',datatype = 'test',num_load=10, trans = trans,trans1=trans_depth)

test_dataloader = DataLoader(testset, batch_size = 1, shuffle=True)

model.eval()
test_loss, correct = 0, 0

device = "cuda" if torch.cuda.is_available() else "cpu"
T = transforms.Resize((240*2, 320*2))


os.makedirs('./out_images', exist_ok = True)
folder = './out_images/'
num = 10
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X.float())
        out = T(pred)

        GT = colorize(y.data,None,None).transpose((0,2,3,1))
        before_norm = colorize(out.data).transpose((0,2,3,1))
        
        #out = 1000/out
        diff = 1000/out - y
        square = diff.cpu().numpy()**2
        n = y.shape[2]*y.shape[3]
        rel = np.sum(np.sqrt(square)/y.cpu().numpy())/n
        rms = np.sqrt(np.sum(square)/n)
        norm = colorize(out.data,None,None).transpose((0,2,3,1))
        
        plt.subplot(2,2,1)
        plt.title('origin')
        plt.imshow(X.detach().cpu().numpy().transpose((0,2,3,1)).squeeze())
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2,2,2)
        plt.title('GT')
        plt.imshow(GT.squeeze())
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2,2,3)
        plt.title('before_norm')
        plt.imshow(before_norm.squeeze())
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2,2,4)
        plt.title('norm')
        plt.imshow(norm.squeeze())
        plt.xticks([])
        plt.yticks([])
        #plt.show()
        save_path = folder + str(num) +'.png'
        plt.savefig(save_path)
        num += 1

# test_accuracy , test_loss = test(test_dataloader, model, cost) 
# print(f"Test Error: \n Accuracy: {(test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")