import os 
import pydicom
import h5py
import numpy as np
#import shutil
import csv
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
import shutil
import argparse
import os



parser = argparse.ArgumentParser()
parser.add_argument("--h5_path_new", type=str)
parser.add_argument("--png_path_new", type=str)
parser.add_argument("--dicom_path_old", type=str)
parser.add_argument("--csv_path", type=str)
parser.add_argument("--num_case", type=int,default=8)
parser.add_argument("--high_val", type=str)
parser.add_argument("--low_val", type=str)
parser.add_argument("--slice_range", type=int,default=2)

#parameters:
#h5_path_new: the path to save generated h5py files
#png_path_new: the path to save generated png files
#dicom_path_old :the path of original dicom path
#num_case: number of cases to user
#high_val: high dose value, example "100" or "75"
#low_val: low dose value, example "6.25" or "50"
#slice_range : how many slices you want to use as the input, add 1 slice range means you add more 2 slices(one above&one below) to the input 



config = parser.parse_args()
NEW_ROOT=config.h5_path_new
NEW_ROOT2=config.png_path_new
NUM_OF_CASE=config.num_case
#from collections import defaultdict
ROOT=config.dicom_path_old
#ORIG_DATA_PATH = '/home/xiangdw/W56 - W56/Pet-Mr Lymphoma/WB-100%-train2/'
#ORIG_DATA_PATH2 = '/home/xiangdw/W56 - W56/Pet-Mr Lymphoma/WB-75%-train2/'
#ORIG_DATA_PATH = '/home/users/anujpare/project_lowhigh/training/'
#NEW_DATA_PATH = '/home/xiangdw/56_100_75_processed2'
#NEW_DATA_PATH = '/home/users/anujpare/project_lowhigh/training_processed/'
OUTPUT_CSV = config.csv_path
if os.path.isdir(NEW_ROOT):
    shutil.rmtree(NEW_ROOT)
if os.path.isdir(NEW_ROOT2):
    shutil.rmtree(NEW_ROOT2)
if not os.path.isdir(NEW_ROOT):
    os.mkdir(NEW_ROOT)
if not os.path.isdir(NEW_ROOT2):
    os.mkdir(NEW_ROOT2)

def get_individual_path(case_num, filename, is_high):
  return f"{NEW_ROOT}/{case_num}_{filename}_{'h' if is_high else 'l'}.hdf5".replace('//','/')
def get_individual_path2(case_num, filename, is_high):
  return f"{NEW_ROOT2}/{case_num}_{filename}_{'h' if is_high else 'l'}.png".replace('//','/')

def get_filename(path):
  filename = path.split('/')[-1].replace('.dcm','')
  filename = filename.replace('IM-0001-','')
  #filename = str(int(filename)).zfill(4)
  return filename

def convert_dicom_to_hdf5_l(path,case_num):
  try:
    #if 'l/' in path:
    #  return
    #case_num = [directory for directory in path.split('/')][-1].split('-')[1]
    new_filename = get_filename(path)
    #print(new_filename)
    #is_high = 'r/' not in path
    is_high= False
    new_path = get_individual_path(case_num, new_filename, is_high)
    new_path2 = get_individual_path2(case_num, new_filename, is_high)
        
    if True:
        dicom_file = pydicom.read_file(path)
        pixel_array = dicom_file.pixel_array
        pixel_array = np.array(pixel_array / 2**4, np.uint8)
        assert np.max(pixel_array) <= 256
        img=pixel_array
        print("std",np.std(pixel_array))
        print("max",np.max(pixel_array))
        print("mean",np.mean(pixel_array))
        if np.std(pixel_array)<=0:
            return None
        else:
            img_mean=np.mean(pixel_array)
            img_std=np.std(pixel_array)
            pixel_array=pixel_array-img_mean
            pixel_array=pixel_array/img_std
            pixel_array = np.array(pixel_array, np.uint8)
            print("type",type(pixel_array[0,0]))
        
        im = Image.fromarray(pixel_array)
        im.save(new_path2)
        h5 = h5py.File(new_path)
        h5.create_dataset(new_filename, data=pixel_array, compression="gzip")
        h5.close()
    return new_path;
  except Exception as e:
    print(e)


def convert_dicom_to_hdf5_h(path,case_num):
  try:
    # if 'l/' in path:
      # return
    #case_num = [directory for directory in path.split('/')][-1].split('-')[1]
    new_filename = get_filename(path)
    #print(new_filename)
    #is_high = 'r/' not in path
    is_high= True
    new_path = get_individual_path(case_num, new_filename, is_high)
    new_path2 = get_individual_path2(case_num, new_filename, is_high)

    if True:
        dicom_file = pydicom.read_file(path)
        pixel_array = dicom_file.pixel_array
        pixel_array = np.array(pixel_array / 2**4, np.uint8)
        assert np.max(pixel_array) <= 256
        
        print("std",np.std(pixel_array))
        print("max",np.max(pixel_array))
        print("mean",np.mean(pixel_array))
        if np.std(pixel_array)<=0:
            return None
        else:
            img_mean=np.mean(pixel_array)
            img_std=np.std(pixel_array)
            pixel_array=pixel_array-img_mean
            pixel_array=pixel_array/img_std
            pixel_array = np.array(pixel_array, np.uint8)
            print("type",type(pixel_array[0,0]))
            
        im = Image.fromarray(pixel_array)
        im.save(new_path2)
        h5 = h5py.File(new_path)
        h5.create_dataset(new_filename, data=pixel_array, compression="gzip")
        h5.close()
    return new_path;
  except Exception as e:
    print(e)


def write_csv(data_path, new_paths_low,new_paths_high):
  pairs = []
  no_pair=[]
  for path in new_paths_high:
    if 'h.hdf5' in path:
      pair_path = path.replace('h.hdf5','l.hdf5')
      is_high=True
    else:
      raise Exception("Bad endings")
    i=-config.slice_range
    med_set=[]
    #print("pair_pth",pair_path)
    if pair_path in new_paths_low:
        #print(pair_path)
        case_num=pair_path.split('/')[-1].split('_')[0]
        #print(case_num)
        slice_num=int(pair_path.split('/')[-1].split('_')[1].split('-')[0])
        for i in range(-config.slice_range,config.slice_range+1,1):       #[-4,-3,-2,-1,0,1,2,3,4]
            med_slice_path=os.path.join(NEW_ROOT,case_num+"_"+str(i+slice_num).zfill(4)+"-0001_l.hdf5")
            #print(med_slice_path)
            if not os.path.exists(med_slice_path):
                i=-config.slice_range
                break
            #print(str(i+slice_num).zfill(4))
            med_set.append(med_slice_path)
    if i==config.slice_range:
            pairs.append((med_set,path))
    else :
            no_pair.append((med_set,path))
  print(len(pairs),len(no_pair))

  with open(OUTPUT_CSV, 'w') as out_file:
    csv_writer = csv.DictWriter(out_file, fieldnames=['low_path', 'high_path'])
    csv_writer.writeheader()
    for pair in pairs:
      row = {}
      row['low_path'], row['high_path'] = pair
      csv_writer.writerow(row)
  print('CSV written!') 
    

def main():
  new_paths_low = set()
  new_paths_high = set()
  for i in [0,1,2,3,4,7,12,13,14,18,19,20,24]:  #range(NUM_OF_CASE):
    NEW_DATA_PATH=os.path.join(NEW_ROOT,str(i))
    ORIG_DATA_PATH0=os.path.join(ROOT,str(i)+'.zip')
    fileList1=os.listdir(ORIG_DATA_PATH0)
    for k in range(len(fileList1)):
      tem_name=fileList1[k]
      tem_path=os.path.join(ORIG_DATA_PATH0,tem_name)
      fileList2=os.listdir(tem_path)
      ORIG_DATA_PATH='0'
      ORIG_DATA_PATH2='0'
      for name1 in fileList2:
        if name1.startswith(config.high_val):
          ORIG_DATA_PATH=os.path.join(tem_path,name1)
          print(ORIG_DATA_PATH)
        if name1.startswith(config.low_val):
          ORIG_DATA_PATH2=os.path.join(tem_path,name1)
          print(ORIG_DATA_PATH2)
    print("2",ORIG_DATA_PATH2)
    if  ORIG_DATA_PATH=="0" or ORIG_DATA_PATH2=="0":
      print("no file for",i)
      continue
    if not os.path.exists(NEW_DATA_PATH):
      os.makedirs(NEW_DATA_PATH)
    #path='E:/data19.7/W56 - W56/Pet-Mr Lymphoma/WB-100% - train/IM-0001-0001-0001.dcm'
    for path in os.listdir(ORIG_DATA_PATH):
      #print(path)
      path = os.path.join(ORIG_DATA_PATH,path)
      #print(path)
      new_path = convert_dicom_to_hdf5_h(path,i)
      #print(new_path)
      if new_path:
        #print(f"{path} saved to {new_path}")
          if new_path in new_paths_high:
              raise Exception("Duplicate path?")
          new_paths_high.add(new_path)  
    for filename in os.listdir(ORIG_DATA_PATH2):
      #print(filename)
      path = os.path.join(ORIG_DATA_PATH2,filename)
      #print('0',path)
      new_path = convert_dicom_to_hdf5_l(path,i)
      if new_path:
        #print(f"{path} saved to {new_path}")
          if new_path in new_paths_low:
              raise Exception("Duplicate path?")
          new_paths_low.add(new_path)  
    #print(new_paths)
    #print(NEW_DATA_PATH)
  write_csv(NEW_DATA_PATH, new_paths_low,new_paths_high)


if __name__ == '__main__':
  main()
