# Based on https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#data
from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import PIL
import scipy.misc
import sys
import shutil

from time import strftime
from tqdm import tqdm



#based on the config to choose different generator model
from unet import UNet as UNet


from utils import get_paths, get_data_mean_std,get_data_loader



output_dir="/home/NanditaStorage/pet_output/"

cur_path=os.path.abspath(sys.argv[0])

# Set random seem for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# Number of channels in the training images. For color images this is 3
nc = 9

# Size of z latent vector (i.e. size of generator input)
nz = 1

# # Size of feature maps in generator
# ngf = 16

# Size of feature maps in discriminator
ndf = 16



#config
parser = argparse.ArgumentParser()
parser.add_argument("--freeze", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--num_folds", type=int, default=5)
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument('--lr_d', type=float, default=0.01)
parser.add_argument('--lr_g', type=float, default=0.01)
parser.add_argument('--twiceG',type=str, default= "False")
parser.add_argument('--activation', type=str, default="tanh")
parser.add_argument('--norm', default=None)
parser.add_argument('--short_cut', type=str, default="True")
parser.add_argument('--dirg', type=str, default=None)
parser.add_argument('--dird', type=str, default=None)
parser.add_argument('--label', type=str, default=None)
parser.add_argument('--lamda', type=float, default=0.05)
parser.add_argument('--lamda2', type=float, default=0.05)
parser.add_argument('--loss', type=str, default="l1")
parser.add_argument('--pool', type=str,default="False")
parser.add_argument('--single_conv_in_down', type=bool, default=False)
parser.add_argument('--num_slice', type=int, default=4)
parser.add_argument("--csv_path", type=str)
parser.add_argument("--train_D", type=str,default="False")
parser.add_argument("--train_G", type=str,default="False")
config = parser.parse_args()



#set the configs
lamda=config.lamda
lamda2=config.lamda2
num_slice=config.num_slice
config.cuda = not config.no_cuda
label=config.label

# Set constants.
log_dir = "./.run_logs" + ("_debug/" if config.debug else "/")
start_date = strftime("%Y_%m_%d")
start_time = strftime("%H_%M_%S")

#save the train.py to output file
config.out_dir = os.path.join(output_dir, f"train_{start_date}_{start_time}_{label}")
if not os.path.exists(config.out_dir):
  os.makedirs(config.out_dir)

print(cur_path)
shutil.copy(cur_path, os.path.join(config.out_dir,'train.py')) 
print(f"Saving results and model to {config.out_dir}")
print(config)

# Get configs
if config.debug:
  print("#" * 80 + "\nWE ARE DEBUGGING\n" + "#" * 80)






# Get the low-dose pet and high-dose pet path through utils.get_path()
low_paths, high_paths = get_paths(config.csv_path,.01 if config.debug else 1.)


#the following command have the same use, but I have not tried 

from sklearn.model_selection import train_test_split
low_path_splits_train, low_path_splits_test, high_path_splits_train, high_path_splits_test= train_test_split( low_paths, high_paths, test_size=0.1, random_state=1)


##
#No transformation no normalization, normalization has already been done when transforming dicom to h5py
"""
normalize = get_data_mean_std(get_data_loader(low_path_splits[0], 
                              high_paths=high_path_splits[0],
                              batch_size=config.batch_size, 
                              shuffle=False, 
                              num_workers=config.num_workers,
                              augment=False, 
                              crop=False,
                             normalize=None,
                              stack=False))
"""
loaders = {
  "train": get_data_loader(low_path_splits_train, 
                           high_paths=high_path_splits_train,
                           batch_size=config.batch_size,
                           shuffle=True, 
                           num_workers=config.num_workers,
                           augment=False, 
                           balance_classes=False,
                           crop=False,
                           normalize=None,#normalize,
                           stack=False,
                           num_slice=num_slice,
                           norm=config.norm,
  ),
  "test":  get_data_loader(high_path_splits_test, 
                           high_paths=high_path_splits_test,
                           batch_size=config.batch_size,
                           shuffle=False, 
                           num_workers=config.num_workers,
                           augment=False, 
                           balance_classes=False,
                           crop=False,
                           normalize=None,
                           stack=False,
                           num_slice=num_slice,
                           norm=config.norm,
  ),
}

# Decide which device we want to run on
device = torch.device("cuda:0")

# custom weights initialization called on netG and netD
def weights_init(m):
    
    #print(m)
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Conv2d):
        #print(m.weight.shape)
        nn.init.normal_(m.weight,0,0.01)
    elif classname.find('Linear') != -1:
        #print(m.weight.shape)
        nn.init.normal_(m.weight,0,0.01)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        #print(m.weight.shape)
        nn.init.normal_(m.weight,0,0.01)
        nn.init.constant_(m.bias.data, 0.0)

# Create the generator
# "act" means different activation method in the last layer
# based on the config to import different UNet
# UNet(self, n_channels, n_classes,act,num_slice):
#parameters:
    #n_channels:input channels
    #n_classes:output channels
    #act: activation method in the last layer. I do this because jiarong suggested that different activation may have impact
    #num_slice: num of slices to use as the input, normally n_channels=2*num_slice+1
if config.activation=="tanh":
    act=F.tanh
elif config.activation=="soft":
    act=F.softplus
elif config.activation=="sigmoid":
    act=F.sigmoid
if config.short_cut== "True":
    if config.pool=="False":
        print("no pool")#"no pool" means no pooling do reduce the size
        netG = UNet_nopool(2*num_slice+1,1,act,config.num_slice).to(device)
    else:
        print("pool")
        netG = UNet(2*num_slice+1,1,act,config.num_slice).to(device)
else:
    netG = UNet_orig(2*num_slice+1,1,act,config.num_slice).to(device)



#netG2 is the second generator, you could decide to do stacked generator or not
netG2=UNet(1,1,act,0).to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)
netG2.apply(weights_init)

# Print the model
# print(netG)


class Flatten(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)

class Discriminator(nn.Module):
  def __init__(self, nc,ndf):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
      # input is (nc) x 64 x 64
      nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf) x 32 x 32
      nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*2) x 16 x 16
      nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 8 x 8
      nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 8),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*8) x 4 x 4
      nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
      # nn.BatchNorm2d(ndf * 8),
      nn.LeakyReLU(0.2, inplace=True),
      Flatten(),
      nn.Linear(81, 1),
      nn.Sigmoid(),
    )

  def forward(self, input):
    return self.main(input)

# Create the Discriminator
netD = Discriminator(1,16).to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
# print(netD)


#based on config to choose different loss function
# Initialize BCELoss function
criterion = nn.BCELoss()
if config.loss =="huberl1":
    criterion2 = nn.SmoothL1Loss()
elif config.loss =="l1":
    criterion2 = nn.L1Loss()
elif config.loss =="l2":
    criterion2 = nn.MSELoss()




fixed_noise = None


# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=config.lr_d, betas=(.5, 0.999),weight_decay=0.0005)
# optimizerG = optim.Adam(netG.parameters(), lr=config.lr_g, betas=(.5, 0.999))
optimizerG = optim.Adam(netG.parameters(),
                        lr=config.lr_g,
                        betas=(.5, 0.999),
                        weight_decay=0.0005)
optimizerG2 = optim.Adam(netG2.parameters(),
                        lr=config.lr_g,
                        betas=(.5, 0.999),
                        weight_decay=0.0005)

scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG,mode='min',factor=0.8)

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []# generator loss
D_losses = []# discriminator loss
L1_losses = []# l1 loss for the generator
D_true = []# output of discriminator on true high dose pet
D_fake = []# output of discriminator on generated high dose pet(fake)
iters = 0

def psnr_c(img1, img2):
#since the max of img1 and img2 may be different.
#I use the max of these images to calculate
    max1= float(np.max(img1))
    max2= float(np.max(img2))
    """
    if max1 < 6.0:
        print("img1 max",max1)
    if max2 < 6.0:
        print("img2 max",max2)
    """
    mse = np.mean((img1/max1 - img2/max2) ** 2 )
    if mse < 1.0e-10:
      return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mse_c(img1, img2):
   max1= float(np.max(img1))
   max2= float(np.max(img2))
   mse = np.mean((img1/max1 - img2/max2) ** 2 )
   return math.sqrt(mse)


print("Starting Training Loop...")
# For each epoch
dataloader = loaders['train']
dataloader_test = loaders['test']

psnr_score=[]
mse_score=[]
max_syn=[]
mean_syn=[]
if config.dirg!=None:
    netG.load_state_dict(torch.load(config.dirg))
if config.dird!=None:
    netD.load_state_dict(torch.load(config.dird))

#if don't use normalize it's 0~1
#if use norm it would be 6.69~ -0.05

#torch norm is the same as sel norm



if not os.path.exists(os.path.join(config.out_dir,"output_images")):
    os.mkdir(os.path.join(config.out_dir,"output_images"))



for epoch in range(config.n_epochs):
  #set l1 loss accumulation to calculate the average loss
  L1_loss_cum=0
  L1_loss_cum_2=0
  loss_ave=0
  
  #For each batch in the dataloader 
  for i, data in enumerate(dataloader, 0):

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0
    
    low, high,_= data
    low=low.float()
    high=high.float()
    if config.cuda:
      low = low.cuda()
      high = high.cuda()
    fake,_ = netG(low)
 
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    
    
    netD.zero_grad()
    # Format batch
    real_cpu = data[0].to(device)
    b_size = real_cpu.size(0)
    label = torch.full((b_size,), real_label, device=device)
    # Forward pass real batch through D
    output = netD(high) #.view(-1)
    # Calculate loss on all-real batch
    #lamda is the parameter to control the ratio of loss between l1 loss and discriminator loss
    errD_real = lamda*criterion(output.view(-1,1), label.view(-1,1))
    # Calculate gradients for D in backward pass
    if config.train_D=="True":
        errD_real.backward()
        #optimizerD.step()
    D_x = output.mean().item()

    ## Train with all-fake batch
    # Generate batch of latent vectors
    # Generate fake image batch with G
    label.fill_(fake_label)
    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = lamda*criterion(output, label)
    # Calculate the gradients for this batch
    if config.train_D=="True":
        errD_fake.backward()
        optimizerD.step()
    D_G_z1 = output.mean().item()
    # Add the gradients from the all-real and all-fake batches
    errD = errD_real + errD_fake
        

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = netD(fake.detach()).view(-1)
    # Calculate G's loss based on this output
    errG = lamda2*criterion(output, label)
    # Calculate gradients for G    
    errG.backward()
    optimizerG.step()
    #using discriminator loss to update G
    
    netG.zero_grad()
    L1_loss=criterion2(fake,high)
    L1_loss.backward()
    D_G_z2 = output.mean().item()   
    optimizerG.step()
    # using la distance between target and output to Update G again
    L1_loss_cum+=L1_loss.item()
    
    #stacked generator to refine first generator's output   
    fake2,_=netG2(fake.detach())
    optimizerG2.zero_grad()
    L1_loss_2=criterion2(fake2,high)
    L1_loss_2.backward()
    # Update G
    optimizerG2.step()  
    L1_loss_cum_2+=L1_loss_2.item()
    
    scheduler.step(L1_loss)
        
    
    # Output training stats
    if i % 100 == 0:
      print(i)
      print(L1_loss_cum)
      l1_loss=L1_loss_cum/(i+1)
      l1_loss_2=L1_loss_cum_2/(i+1)
      print('[%2d/%2d][%4d/%4d] Loss_D: %.4f Loss_G: %.4f L1_Loss: %.4f L1_Loss_2: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
          % (epoch+1, config.n_epochs, i, len(dataloader),
           errD.item(), errG.item(),l1_loss, l1_loss_2,D_x, D_G_z1, D_G_z2))
      # Save Losses for plotting later
      G_losses.append(errG.item())
      D_losses.append(errD.item())
      L1_losses.append(l1_loss)
      D_true.append(D_x)
      D_fake.append(D_G_z1)

    
    iters += 1
  txt=open(os.path.join(config.out_dir,'G_loss.txt'),'w')
  txt.write(str(G_losses))
  txt.close()
  txt=open(os.path.join(config.out_dir,'D_loss.txt'),'w')
  txt.write(str(D_losses))
  txt.close()
  txt=open(os.path.join(config.out_dir,'L1_loss.txt'),'w')
  txt.write(str(L1_losses))
  txt.close()
  txt=open(os.path.join(config.out_dir,'D_true.txt'),'w')
  txt.write(str(D_true))
  txt.close()
  txt=open(os.path.join(config.out_dir,'D_fake.txt'),'w')
  txt.write(str(D_fake))
  txt.close()
  #save the model
  if epoch % 2 ==0:
    torch.save(netD.state_dict(),os.path.join(config.out_dir,'6.25_'+str(epoch)+ '_netD_params.pkl'))
    torch.save(netG.state_dict(),os.path.join(config.out_dir,'6.25_'+str(epoch)+ '_netG_params.pkl'))
    torch.save(netG2.state_dict(),os.path.join(config.out_dir,'6.25_'+str(epoch)+ '_netG2_params.pkl'))
    
  #testing part in each epoch
  num=0
  psnr_ave=0 
  psnr_ave_m=0
  psnr_ave2=0
  psnr_ave3=0
  psnr_ave_l1=0
  psnr_ave_l2=0
  psnr_ave_l3=0
  psnr_ave_l4=0
  psnr_ave_l5=0
  psnr_ave_l6=0
  psnr_ave_l7=0
  mse_ave=0
  for i, data in enumerate(dataloader_test, 0):
    low, high, img_name = data
    if config.cuda:
      low = low.cuda()
      high = high.cuda()   
    #print("test low shape",low.shape)
    low=low.float()
    high=high.float()    
    fake0,low2 = netG(low)
    
    #low2=low,fake0 is the first time output of generator
    
    fake,_=netG2(fake0.detach())
    
    #fake is the refined result from the second generator
    
    fake=fake.cpu().detach().numpy()
    fake0=fake0.cpu().detach().numpy()
    low=low.cpu().detach().numpy()
    low2=low2.cpu().detach().numpy()
    high=high.cpu().detach().numpy()
    # each slice in the batch
    for j in range(high.shape[0]): 
        if fake.shape[1]!=1 or high.shape[1]!=1: #or low.shape[1]!=1:
            print("dimension wrong")  
        name=img_name[j]
        true=high[j,:,:,:].reshape(high.shape[2],high.shape[3])
        max_true=np.max(true)
        false=fake[j,:,:,:].reshape(fake.shape[2],fake.shape[3])
        false_m=fake0[j,:,:,:].reshape(fake0.shape[2],fake0.shape[3])
        low_true=low[j,config.num_slice,:,:].reshape(low.shape[2],low.shape[3])
        #low_true 1~6 seperately means the the 3 slices above and below the center one
        low_true1=low[j,config.num_slice-1,:,:].reshape(low.shape[2],low.shape[3])
        low_true2=low[j,config.num_slice+1,:,:].reshape(low.shape[2],low.shape[3])   
        if config.num_slice>=2:
            low_true3=low[j,config.num_slice-2,:,:].reshape(low.shape[2],low.shape[3])
            low_true4=low[j,config.num_slice+2,:,:].reshape(low.shape[2],low.shape[3]) 
        if config.num_slice>=3:
            low_true5=low[j,config.num_slice-3,:,:].reshape(low.shape[2],low.shape[3])
            low_true6=low[j,config.num_slice+3,:,:].reshape(low.shape[2],low.shape[3])             
        low2_true=low2[j,:,:,:].reshape(low2.shape[2],low2.shape[3])
        # save the test images in the last two epoch or the middle epoch
        if epoch == config.n_epochs-2 or epoch==int(config.n_epochs/2):
            print(epoch)
            vutils.save_image(torch.tensor(low_true),os.path.join(os.path.join(config.out_dir,"output_images"),"input_"+name+'.png'),normalize=True)
            vutils.save_image(torch.tensor(true),os.path.join(os.path.join(config.out_dir,"output_images"),"target_"+name+'.png'),normalize=True)
            vutils.save_image(torch.tensor(false),os.path.join(os.path.join(config.out_dir,"output_images"),"output_"+name+'.png'),normalize=True)          
        psnr=psnr_c(false,true)
        psnr_m=psnr_c(false_m,true)
        psnr2=psnr_c(low_true,true)
        psnrl1=psnr_c(low_true1,true)
        psnrl2=psnr_c(low_true2,true)
        psnrl3=0
        psnrl4=0
        psnrl5=0
        psnrl6=0
        if config.num_slice>=2:
            psnrl3=psnr_c(low_true3,true)
            psnrl4=psnr_c(low_true4,true)
        if config.num_slice>=3:
            psnrl5=psnr_c(low_true5,true)
            psnrl6=psnr_c(low_true6,true)
        psnr3=psnr_c(low2_true,true)
        mse=mse_c(false,true)
        if max_true>0.1: #and psnrl3>1 and psnrl4>1:
            psnr_ave=(psnr_ave*num+psnr)/(num+1)
            psnr_ave_m=(psnr_ave_m*num+psnr_m)/(num+1)
            psnr_ave2=(psnr_ave2*num+psnr2)/(num+1)
            psnr_ave3=(psnr_ave3*num+psnr3)/(num+1)
            psnr_ave_l1=(psnr_ave_l1*num+psnrl1)/(num+1)
            psnr_ave_l2=(psnr_ave_l2*num+psnrl2)/(num+1)
            psnr_ave_l3=(psnr_ave_l3*num+psnrl3)/(num+1)
            psnr_ave_l4=(psnr_ave_l4*num+psnrl4)/(num+1)
            psnr_ave_l5=(psnr_ave_l5*num+psnrl5)/(num+1)
            psnr_ave_l6=(psnr_ave_l6*num+psnrl6)/(num+1)
            mse_ave=(mse_ave*num+mse)/(num+1)
            num +=1
  print("psnr_ave_output_with_two_generator",psnr_ave)
  print("psnr_ave_output_with_one_generator",psnr_ave_m)
  print("psnr_ave_low_dose",psnr_ave2)
  print("psnr_ave_lower_one",psnr_ave_l1)
  print("psnr_ave_higher_one",psnr_ave_l2)
  print("psnr_ave_lower_two",psnr_ave_l3)
  print("psnr_ave_higher_two",psnr_ave_l4)
  print("psnr_ave_lower_three",psnr_ave_l5)
  print("psnr_ave_higher_three",psnr_ave_l6)
  print("mse_ave",mse_ave)
  psnr_score.append(psnr_ave)
  mse_score.append(mse_ave)
  txt=open(os.path.join(config.out_dir,'psnr_result.txt'),'w')
  txt.write(str(psnr_score))
  txt.close()
  txt=open(os.path.join(config.out_dir,'mse_result.txt'),'w')
  txt.write(str(mse_score))
  txt.close()





txt=open(os.path.join(config.out_dir,'config.txt'),'w')
txt.write(str(config))
txt.close()


