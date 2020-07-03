import json

with open("data.json",'r') as load_f:
  load_dict = json.load(load_f)
  print(load_dict.keys())
  #print(load_dict['envs'])
  print(load_dict['val_env_idx'])
  print(load_dict['envs'][395])
