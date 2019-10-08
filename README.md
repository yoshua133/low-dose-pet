# low-dose-pet
This for generating 100% dose PET images based on low dose images.
Dataset consists of 40 cases of scans. Each case has 400~600 slices.

## preprocess  

To do the preprocessing, prepare a dicom dataset first. The dataset need to be named from 0 to [num of cases]. And for each case, there has to be 2 folders of high dose and low dose.
Then run `preprocess.py` . This would create a folder of h5py files and another folder of png files.
Following command like this:
  

    python l2h/preprocess2.py $@ --h5_path_new "/home/dxiang/unzipped_6.25_100_low_val_cases" --png_path_new    "/home/dxiang/unzipped_6.25_100_low_val_cases_png" --dicom_path_old "/home/dxiang/total_images/" --csv_path   "/home/dxiang/unzipped_6.25_100_low_val_cases/csv.csv" --num_case 8 --high_val "100" --low_val "6.25" --slice_range 4
    
This would create a png file folder and a h5py file folder in the foleder name "png_path_new" and "h5_path_new", meanwhile doing the normalization on each image.
  
## training
  
To do the training, run `main.py` like this
  

    CUDA_VISIBLE_DEVICES=1 python l2h/main.py $@ --n_epochs=25 --lr_d=.005 --lr_g=.005 --label="debug" --train_D="True" --twiceG="False" --lamda=0.05 --lamda2=0.05 \
    --activation="tanh" --loss="l2" --pool="True" --num_slice=3 --batch_size=32 --csv_path="/home/dxiang/unzipped_6.25_100_low_val_cases/csv.csv"
