# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import cv2
import time
import argparse
import numpy as np
import os, sys, json

import torch
from torch.autograd import Variable
torch.backends.cudnn.enabled = False
import torch.multiprocessing as mp

from models import VqaLstmModel, VqaLstmCnnAttentionModel
from data_angle_multi import EqaDataset, EqaDataLoader
from metrics import VqaMetric

from models import get_state, repackage_hidden, ensure_shared_grads
from data_angle_multi import load_vocab

import pdb
import neural_renderer as nr
import math

import json
import csv

def handle_load(trained_model_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in trained_model_dict.items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v
    return new_state_dict

def handle_file(path):
    return 

def find_index(name, house):
    rf_o = open('/media/trs1/dataset/suncg_data/house/' + house + '/house.obj')
    idx = 0
    s = 0
    for line in rf_o:
        line_s = line.split(' ')[0]
        if(line_s == 'g' and s==1):#FIRST LAST
            print(line)
            end = idx
            print(end)
            rf_o.close()
            return begin, end
        if(line == 'g Model#' + name +'\n'):#FIRST LAST
            print(line)
            begin = idx
            print(begin)
            s = 1
        if(line_s == 'f'):
            idx = idx + 1
    rf_o.close()
    begin = 1000
    end = 2000
    return begin, end

def find_index_v(name, house):
    rf_o = open('/media/trs1/dataset/suncg_data/house/' + house + '/house.obj')
    idx = 0
    s = 0
    vlist = []
    vpos = 1
    for line in rf_o:
        words = line.split()
        line_s = line.split(' ')[0]
        if(line_s == 'g' and s==1):#FIRST LAST
            print(line)
            end = idx
            print(end)
            rf_o.close()
            return list(set(vlist))
        if(line == 'g Model#' + name +'\n'):#FIRST LAST
            print(line)
            begin = idx
            print(begin)
            s = 1
        if(line_s == 'f' and s==1):
            vword = words[vpos].split('/')
            for pos in vword:
                if(pos != ''):
                    vlist.append(int(pos))
    rf_o.close()
    begin = 1000
    end = 2000
    return None
def get_info_v(idx, house):
    with open('../orieqa_v1.json') as f:
        d = json.load(f)
    d = d['questions'][house]
    for i in range(len(d)):
        id = d[i]['id']
        if(id == idx):
            name = d[i]['bbox'][0]['name'].split('/')[0]
            break
    num = []
    lines = csv.reader(open('/home/dsg/xuyitao/neuEmbodiedQA1/House3D/House3D/metadata/ModelCategoryMapping.csv','r'))
    for line in lines:
        if(line[2] == name):
            num.append(line[1])
    with open('/media/trs1/dataset/suncg_data/house/' + house + '/house.json') as f:
        d = json.load(f)
    for i in range(len(num)):
        name = num[i]
        for j in range(len(d['levels'])):
            nodes = d['levels'][j]['nodes']
            for k in range(len(nodes)):
                if(d['levels'][j]['nodes'][k]['modelId'] == name):
                    print(name)
                    return find_index_v(name, house) # not break
    return find_index_v(name, house)
def get_info(idx, house):
    with open('../orieqa_v1.json') as f:
        d = json.load(f)
    d = d['questions'][house]
    for i in range(len(d)):
        id = d[i]['id']
        if(id == idx):
            name = d[i]['bbox'][0]['name'].split('/')[0]
            break
    num = []
    lines = csv.reader(open('/home/dsg/xuyitao/neuEmbodiedQA1/House3D/House3D/metadata/ModelCategoryMapping.csv','r'))
    for line in lines:
        if(line[2] == name):
            num.append(line[1])
    with open('/media/trs1/dataset/suncg_data/house/' + house + '/house.json') as f:
        d = json.load(f)
    for i in range(len(num)):
        name = num[i]
        for j in range(len(d['levels'])):
            nodes = d['levels'][j]['nodes']
            for k in range(len(nodes)):
                if(d['levels'][j]['nodes'][k]['modelId'] == name):
                    print(name)
                    return find_index(name, house) # not break
    return find_index(name, house)
def fgsm(rank, args, shared_model, number):

    torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))
    all_n = 0

    # torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))

    model_kwargs = {'vocab': load_vocab(args.vocab_json)}
    model = VqaLstmCnnAttentionModel(**model_kwargs)
    device_ids = [0,1]

    model = model.cuda(device_ids[0])
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    # torch.backends.cudnn.benchmark = True
    
    lossFn = torch.nn.CrossEntropyLoss().cuda()

    eval_loader_kwargs = {
        'questions_h5': getattr(args, args.eval_split + '_h5'),
        'data_json': args.data_json,
        'vocab': args.vocab_json,
        'target_obj_conn_map_dir': args.target_obj_conn_map_dir,
        'batch_size': args.batch_size,
        'input_type': args.input_type,
        'num_frames': args.num_frames,
        'split': args.eval_split,
        'max_threads_per_gpu': args.max_threads_per_gpu,
        'gpu_id': args.gpus[rank%len(args.gpus)],
        'to_cache': args.cache
    }

    eval_loader = EqaDataLoader(**eval_loader_kwargs)
    # for ijcai in range(number):
    #     eval_loader.dataset._load_envs()
    eval_loader.dataset._load_envs(start_idx=number)
    print('eval_loader has %d samples' % len(eval_loader.dataset))

    args.output_log_path = os.path.join(args.log_dir,
                                        'eval_' + str(rank) + '.json')

    model.load_state_dict(handle_load(shared_model.state_dict()))
    model.eval()

    metrics = VqaMetric(
        info={'split': args.eval_split},
        metric_names=[
            'loss', 'accuracy', 'mean_rank', 'mean_reciprocal_rank'
        ],
        log_json=args.output_log_path)


    all_envs_loaded = eval_loader.dataset._check_if_all_envs_loaded()
    done = False
    print(number, ' begin')
    import copy
    import torch.nn as nn
    from PIL import Image
    softmax = nn.Softmax()
    allcor = []
    while done == False:
        for ii,batch in enumerate(eval_loader):
            # model.cuda()
            if(ii>=0):
                idx, questions, answers, house, v, f, vt, pos, _, _, _ = batch
                print('all size:',v.size(),f.size(),vt.size())
                #print(house)
                print(questions, answers)
                questions_var = Variable(questions.cuda())
                print(questions_var.size())
                answers_var = Variable(answers.cuda())
                v_var = Variable(v.cuda(),requires_grad=True)
                f_var = Variable(f.cuda())
                vt_var = Variable(vt.cuda(),requires_grad=True)
                begin, end = get_info(idx[0], house[0])
                #print(vt_var[0][0][40010][0][0])  
                #print(vt_var[0][0][39506][0][0]) 
                vt_test = copy.deepcopy(vt_var)
                epsilon = 32.0/256.0
                vt_grad = torch.zeros(vt_var.size()).cuda()
                    #vt_var.retain_grad()
                scores, att_probs,img_clean = model(v_var, f_var, vt_var, pos, questions_var)
                i1, i2 = torch.max(scores[0],0)
                mi = i2.cpu().numpy().tolist()
                ms = int(answers)
                print(mi)
                print(mi==int(ms))
                if(mi==int(ms)):
                    allcor.append(1.0)
                else:
                    allcor.append(0.0)
                print(softmax(scores[0]))
                print(softmax(scores[0])[ms])
                img_clean = img_clean[0]
                for iii in range(img_clean.size()[0]):
                    imggg = img_clean[iii].detach().cpu().numpy()
                    imggg = imggg * 255.0
                    imggg = imggg.transpose((1,2,0))
                    imggg = Image.fromarray(imggg.astype('uint8'))
                    imggg.save('result_test/'+str(ii)+'_'+str(iii)+'_clean.jpg')
                loss = lossFn(scores, answers_var)
                loss.backward()
                #print(torch.max(vt_grad))
                vt_grad = vt_var.grad
                v_grad = v_var.grad
                print('max grad',torch.max(vt_grad.data))
                #print(vt_grad[0][0][40010][0][0])
                #print(vt_grad[0][0][39506][0][0])
                vt_var = vt_var.detach() + epsilon * torch.sign(vt_grad)
                #v_var = v_var.detach() + 1.0 * torch.sign(v_grad)
                vt_var = torch.clamp(vt_var, 0, 1)
                #vt_var = Variable(vt_var.data, requires_grad=True).cuda()
                with torch.no_grad():
                    model.eval()
                    begin, end = get_info(idx[0], house[0]) 
                    #for iii in range(begin,end):
                        #if(vt_test[0][0][iii][0][0][0][0] != vt_var[0][0][iii][0][0][0][0] or vt_test[0][0][iii][0][0][1][0] != vt_var[0][0][iii][0][0][1][0] or vt_test[0][0][iii][0][1][1][0] != vt_var[0][0][iii][0][1][1][0]):
                            #print(iii)
                    vt_test[0][0][begin:end] = vt_var[0][0][begin:end]
                    #print(vt_test[0][0][40010][0][0])
                    #print(vt_test[0][0][39506][0][0])
                    #print((vt_test[0][0] == vt_var[0][0]).sum())
                    #print((vt_test[0][0].size()),(vt_var[0][0].size()))
                    scores, att_probs,imgg = model(v_var, f_var, vt_test, pos, questions_var)
                    imgg = imgg[0]
                    #print(imgg.size())
                    for iii in range(imgg.size()[0]):
                        imggg = imgg[iii].detach().cpu().numpy()
                        imggg = imggg * 255.0
                        imggg = imggg.transpose((1,2,0))
                        imggg = Image.fromarray(imggg.astype('uint8'))
                        imggg.save('result_test/'+str(ii)+'_'+str(iii)+'_adv.jpg')
                    i1, i2 = torch.max(scores[0],0)
                    mi = i2.cpu().numpy().tolist()
                    ms = int(answers)
                    print(mi)
                    print(mi==int(ms))
                    print(softmax(scores[0]))
                    print(softmax(scores[0])[ms])
                for k in range(idx.shape[0]):
                    begin, end = get_info(idx[k], house[k]) 
                    #print(begin, end)
                    #begin, end = find_index('167', house[k]) 
                    v_m = v_var[k][0]
                    f_m = f_var[k][0][begin:end]
                    vt_m = vt_var[k][0][begin:end]
                    nr.save_obj('/media/trs1/dataset/suncg_data/house/' + house[k] + '/attack_' + str(int(idx[k])) + '.obj', v_m, f_m, vt_m)
                    idx = idx.cpu()
                    questions = questions.cpu()
                    answers = answers.cpu()
                    questions_var = questions_var.cpu()
                    answers_var = answers_var.cpu()
                    v = v.cpu()
                    f = f.cpu()
                    vt = vt.cpu()
                    v_m = v_m.cpu()
                    f_m = f_m.cpu()
                    vt_m = vt_m.cpu()
                    v_var = v_var.detach().cpu()
                    f_var = f_var.cpu()
                    vt_var = vt_var.detach().cpu()
                    vt_grad = vt_grad.cpu()
                    print(house[k] + ' ' + str(int(idx[k])) + ' ok')
                    all_n += 1
                    # handle_file(path)

            if all_envs_loaded == False:
                eval_loader.dataset._load_envs()
                if len(eval_loader.dataset.pruned_env_set) == 0:
                    done = True
            else:
                done = True
        print(allcor)
        print(number, ' over')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('-train_h5', default='data/train.h5')
    parser.add_argument('-val_h5', default='data/val.h5')
    parser.add_argument('-test_h5', default='data/test.h5')
    parser.add_argument('-data_json', default='data/data.json')
    parser.add_argument('-vocab_json', default='data/vocab.json')

    parser.add_argument('-train_cache_path', default=False)
    parser.add_argument('-val_cache_path', default=False)

    parser.add_argument('-mode', default='attack', type=str, choices=['train','eval','attack'])
    parser.add_argument('-eval_split', default='test', type=str)

    # model details
    parser.add_argument(
        '-input_type', default='ques,image', choices=['ques', 'ques,image'])
    parser.add_argument(
        '-num_frames', default=5,
        type=int)  # -1 = all frames of navigation sequence

    # optim params
    parser.add_argument('-batch_size', default=1, type=int)
    parser.add_argument('-learning_rate', default=3e-4, type=float)
    parser.add_argument('-max_epochs', default=1000, type=int)

    # bookkeeping
    parser.add_argument('-print_every', default=50, type=int)
    parser.add_argument('-eval_every', default=1, type=int)
    parser.add_argument('-identifier', default='ques-image')
    parser.add_argument('-num_processes', default=1, type=int)
    parser.add_argument('-max_threads_per_gpu', default=10, type=int)
    parser.add_argument(
        '-target_obj_conn_map_dir',
        default=False)
    # checkpointing
    parser.add_argument('-checkpoint_path', default=False)
    parser.add_argument('-checkpoint_dir', default='checkpoints/vqa/')
    parser.add_argument('-log_dir', default='logs/vqa/')
    parser.add_argument('-log', default=False, action='store_true')
    parser.add_argument('-cache', default=False, action='store_true')
    args = parser.parse_args()

    args.time_id = time.strftime("%m_%d_%H:%M")

    #os.environ['CUDA_VISIBLE_DEVICES']='1,2'
    try:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        args.gpus = [int(x) for x in args.gpus]
    except KeyError:
        print("CPU not supported")
        exit()

    if args.checkpoint_path != False:
        print('Loading checkpoint from %s' % args.checkpoint_path)

        args_to_keep = ['input_type', 'num_frames']

        checkpoint = torch.load(args.checkpoint_path, map_location={'cuda:0': 'cpu'})

        for i in args.__dict__:
            if i not in args_to_keep:
                checkpoint['args'][i] = args.__dict__[i]

        args = type('new_dict', (object, ), checkpoint['args'])

    args.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                       args.time_id + '_' + args.identifier)
    args.log_dir = os.path.join(args.log_dir,
                                args.time_id + '_' + args.identifier)

    print(args.__dict__)

    if not os.path.exists(args.checkpoint_dir) and args.log == True:
        os.makedirs(args.checkpoint_dir)
        os.makedirs(args.log_dir)

    model_kwargs = {'vocab': load_vocab(args.vocab_json)}
    shared_model = VqaLstmCnnAttentionModel(**model_kwargs)
    if args.checkpoint_path != False:
        print('Loading params from checkpoint: %s' % args.checkpoint_path)
        shared_model.load_state_dict(checkpoint['state'])
    shared_model.share_memory()
    fgsm(0, args, shared_model, 0)
    torch.cuda.empty_cache()










