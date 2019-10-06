import sys
sys.path.append('.')
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils
import h5py
import csv
import threading
import queue
import math
import random
random.seed(999)
from PIL import Image as pil_image
from PIL import ImageOps
from tqdm import tqdm
from collections import Counter

#from l2h.transformations import transform_image
from constants import INTERPOLATION, PIL_INTERPOLATION_METHODS
#num_slice=3
# Prevents a crash.
torch.multiprocessing.set_sharing_strategy('file_system') 

def combine_low_high(low_paths, high_paths):
  paths = low_paths + high_paths
  labels = ([1] * len(low_paths)) + ([2] * len(high_paths))
  return paths, labels

def get_paths(csv_path,random_keep=1.0): 
  print(csv_path)
  low_paths, high_paths = [], []
  with open(csv_path, mode="r") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    rows = list(csv_reader)
    for row in rows:
      if random.random() < random_keep:
        low_paths.append(row["low_path"])
        high_paths.append(row["high_path"])
    
  return low_paths, high_paths

def get_default_mri_loader(input_size):
  # default image loader from hdf5
  def default_mri_loader(path,num_slice,islow=False):
    if islow:
      #print(path)
      path=path.strip('[')
      path=path.strip(']')
      path=path.replace("'","")
      path=path.replace(" ","")
      #print(path)
      path_set=path.split(',')
      #print(path_set[0])
      img=np.zeros([2*num_slice+1,192,192])
      for i in range(2*num_slice+1):
        img_path=path_set[4-num_slice+i]
        img[i,:,:]=np.array(load_img(img_path),np.float)
      #print(i)
      #print('get_def',img.shape)
      img_t=img
      #print("get def max",np.max(img_t))
      return img
    else:
      img = load_img(
        path, 
        color_mode="grayscale", 
        target_size=(input_size, input_size, 1) if input_size else None, 
        interpolation=INTERPOLATION,
      )
      img=img.reshape(1,img.shape[0],img.shape[1])
    img_t=np.array(img)
    
    return img

  return default_mri_loader
  
def get_data_loader(
  low_paths, 
  high_paths=None, 
  labels=None,
  input_size=None,
  batch_size=32, 
  shuffle=False,
  num_workers=4,
  balance_classes=False,
  augment=False,
  crop=False,
  normalize=None,
  stack=True,
  norm=None,
  num_slice=3
):
  # Load front image index
  dataset = MRIFileList(
      low_paths=low_paths,
      high_paths=high_paths,
      labels=labels,
      loader=get_default_mri_loader(input_size=input_size),
      num_workers=num_workers,
      norm=norm,
      num_slice=num_slice
    )
  # Build data loader
  data_loader = torch.utils.data.dataloader.DataLoader(
    dataset,
    sampler=None,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
  )
  
  return data_loader 

def load_img(path, color_mode="grayscale", target_size=None, interpolation="bicubic"):
  """Loads an image into PIL format.
  # Arguments
    path: Path to image file.
    color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
      The desired image format.
    target_size: Either `None` (default to original size)
      or tuple of ints `(img_height, img_width)`.
    interpolation: Interpolation method used to resample the image if the
      target size is different from that of the loaded image.
      Supported methods are "nearest", "bilinear", and "bicubic".
      If PIL version 1.1.3 or newer is installed, "lanczos" is also
      supported. If PIL version 3.4.0 or newer is installed, "box" and
      "hamming" are also supported. By default, "nearest" is used.
  # Returns
    A PIL Image instance.
  # Raises
    ImportError: if PIL is not available.
    ValueError: if interpolation method is not supported.
  """
  # h5 = h5py.file(path)
  # img = h5.get(path.split('.')[0])
  # return img
  if pil_image is None:
    raise ImportError(
      "Could not import PIL.Image. " "The use of `array_to_img` requires PIL."
    )
  h5 = h5py.File(path, 'r')
  filename = path.replace('.hdf5','').split('/')[-1].split('_')[1]
  img = h5.get(filename)
  img = np.array(img,np.float)
  return img



    
class MRIFileList(torch.utils.data.Dataset):
  def __init__(self, 
         low_paths, 
         norm,
         num_slice,
         high_paths=None,
         labels=None,
         loader=None,
         transform=None, 
         num_workers=8
         ):
    self.low_paths = low_paths
    self.high_paths = high_paths
    self.labels = labels
    self.loader = loader
    self.cache = {}
    self.norm=norm
    self.num_slice=num_slice
# transform would convert it to 0~1
  def __getitem__(self, index):
    low_path = self.low_paths[index]
    if low_path not in self.cache:
      self.cache[low_path] = self.loader(low_path,self.num_slice,islow=True)
    low_img = self.cache[low_path].copy()
    #print('before_trans_get_item',low_img.shape)
      #print("before trans get item max",np.max(img_t))
    #if self.transform:
    #  low_img = self.transform(low_img)
    #  normlize img to 0~1
    if np.max(low_img)==255:
        low_img=torch.div(low_img , 255.0)
    if self.norm!=None:
        low_img=torch.sub(low_img , self.norm[0])
        low_img=torch.div(low_img , self.norm[1])
    #print('aftere_trans_get_item',low_img.shape)
    
    if self.high_paths:
      high_path = self.high_paths[index]
      if high_path not in self.cache:
        self.cache[high_path] = self.loader(high_path,self.num_slice)
      high_img = self.cache[high_path].copy()
      img_t=np.array(high_img)
      if np.max(high_img)==255:
        high_img=torch.div(high_img , 255.0)
      if self.norm!=None:
        high_img=torch.sub(low_img , self.norm[0])
        high_img=torch.div(low_img , self.norm[1])
      img_t=np.array(high_img)
      name_set=low_path.split("/")[-1].replace(".hdf5","").split("_")
      path_name=name_set[0]+"_"+name_set[1].split("-")[0]
      return torch.tensor(low_img), torch.tensor(high_img),path_name

    if self.labels:
      label = self.labels[index]
      return low_img, label

    return low_img

  def __len__(self):
    return len(self.low_paths)

def get_data_mean_std(dataloader):
  print("Getting data mean and standard deviation.")
  curr_sum = 0
  curr_sum_squares = 0
  for img, _ in tqdm(dataloader):
    img = np.array(img)
    curr_sum += np.mean(img)
    curr_sum_squares += np.mean(np.square(img))

  mean = curr_sum / len(dataloader)
  mean_sq = curr_sum_squares / len(dataloader)
  std = math.sqrt(mean_sq - mean ** 2)
  print(f"Data mean: {mean}, stddev: {std}")
  return mean, std

if __name__ == '__main__':
  low_paths, high_paths = get_paths()
  paths = low_paths + high_paths
  data = get_data_loader(paths)
  for datum in data:
    print(datum.mean())

  