import h5py

f = h5py.File('train.h5','r')   #打开h5文件
# for key in f.keys():
    # print(f[key].name)
    # print(f[key].shape)
    # print(f[key].value)
print(f['idx'].value.tolist().index(117541))
f.close()
 