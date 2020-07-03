import h5py

f = h5py.File('train.h5','r')   #打开h5文件  
for key in f.keys():
  print(key)
f.close()  
