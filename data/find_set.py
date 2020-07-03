f = h5py.File('train.h5')
f['idx'].value.tolist().index(154654)
f.close()

with open('newdata.json') as f:
   d = json.load(f)

