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
from data import EqaDataset, EqaDataLoader
from metrics import VqaMetric

from models import get_state, repackage_hidden, ensure_shared_grads
from data import load_vocab

import pdb
import neural_renderer as nr
import math
import csv

import json
import csv
import numpy as np
import pickle
import copy
                        

def find_index(name, house):
    rf_o = open('/path/to/data/house/' + house + '/house.obj')
    idx = 0
    s = 0
    oid = rid = 0
    for line in rf_o:
        # g Object#0_95
        if(line.split('#')[0] == 'g Object'):
            rid = line.split('_')[-1]
        line_s = line.split(' ')[0]
        if(line_s == 'g' and s==1):#FIRST LAST
            end = idx
            rf_o.close()
            return begin, end, oid
        if(line == 'g Model#' + name +'\n'):#FIRST LAST
            begin = idx
            s = 1
            oid = rid.split('\n')[0]
        if(line_s == 'f'):
            idx = idx + 1
    rf_o.close()
    begin = 1000
    end = 2000
    oid = 0
    return begin, end, oid




def attack_fgsm(rank, args, shared_model, number): #?

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
    while done == False:
        for batch in eval_loader:
            idx, questions, answers, house, v, f, vt, pos, _, _, _ = batch
            questions_var = Variable(questions.cuda())
            answers_var = Variable(answers.cuda())
            v_var = Variable(v.cuda(),requires_grad=True)
            f_var = Variable(f.cuda())
            vt_var = Variable(vt.cuda(),requires_grad=True)   

            # noise level
            epsilon = 12.0 / 255.0
            scores, att_probs = model(v_var, f_var, vt_var, pos, questions_var, 0, '0')
            loss = lossFn(scores, answers_var)
            loss.backward()

            # get grad for attack
            vt_grad = vt_var.grad
            vt_detach = vt_var.detach()
            
            begin, end, oid = get_info(idx[0], house[0]) 
            if(begin==1000 and end==2000):
                print(str(int(idx[0])), 'error')
            vt_grad[0][0][:begin] = 0
            vt_grad[0][0][end:] = 0
            vt_var = vt_detach + epsilon * torch.sign(vt_grad) 
            for k in range(idx.shape[0]):
                begin, end, oid = get_info(idx[k], house[k]) 
                v_m = v_var[k][0]
                f_m = f_var[k][0][begin:end]
                vt_m = vt_var[k][0][begin:end]
                # save changed object to .obj file
                nr.save_obj('/path/to/data/house/' + house[k] + '/attack_' + str(int(idx[k])) + '_' + str(oid) + '.obj', v_m, f_m, vt_m)
            with torch.no_grad():
                model.eval()
                scores, att_probs = model(v_var, f_var, vt_var, pos, questions_var, 0, '0')
                accuracy, ranks = metrics.compute_ranks(scores.data.cpu(), answers)
                
        if all_envs_loaded == False:
            eval_loader.dataset._load_envs()
            if len(eval_loader.dataset.pruned_env_set) == 0:
                done = True
        else:
            done = True

def attack_pgd(rank, args, shared_model, number):

    torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))
    all_n = 0

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
        'batch_size': args.batch_size,
        'input_type': args.input_type,
        'num_frames': args.num_frames,
        'split': args.eval_split,
        'max_threads_per_gpu': args.max_threads_per_gpu,
        'gpu_id': args.gpus[rank%len(args.gpus)],
        'to_cache': args.cache
    }

    eval_loader = EqaDataLoader(**eval_loader_kwargs)
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

    while done == False:
        for batch in eval_loader:
            # model.cuda()
            idx, questions, answers, house, v, f, vt, pos, _, _, _ = batch
		
            questions_var = Variable(questions.cuda())
            answers_var = Variable(answers.cuda())
            v_var = Variable(v.cuda(),requires_grad=True)
            f_var = Variable(f.cuda())
            vt_var = Variable(vt.cuda(),requires_grad=True) 

            attack_iter = 10
            attack_momentum = 1
            alpha = 2.0 / 255 
            epsilon = 16.0 / 255
            vt_grad = torch.zeros(vt_var.size()).cuda()
            for j in range(attack_iter):
                scores, att_probs = model(v_var, f_var, vt_var, pos, questions_var, j, str(int(idx[0])))
                
                loss = lossFn(scores, answers_var)
                loss.backward()           
                vg = vt_var.grad
                
                begin, end, _ = get_info(idx[0], house[0]) 
                if(begin==1000 and end==2000):
                    print(str(int(idx[0])), 'error')
                vg[0][0][:begin] = 0
                vg[0][0][end:] = 0
                            
                noise = attack_momentum * vt_grad + vg
                vt_grad = noise
                vt_var = vt_var.detach() + alpha * torch.sign(noise)
                vt_var = torch.where(vt_var > vt+epsilon, vt+epsilon, vt_var)
                vt_var = torch.clamp(vt_var, 0, 1)
                vt_var = torch.where(vt_var < vt-epsilon, vt-epsilon, vt_var)
                vt_var = torch.clamp(vt_var, 0, 1)
                vt_var = Variable(vt_var.data, requires_grad=True).cuda()

            with torch.no_grad():
                model.eval()
                scores, att_probs = model(v_var, f_var, vt_var, pos, questions_var, 100, str(int(idx[0])))
                accuracy, ranks = metrics.compute_ranks(scores.data.cpu(), answers)
            
            begin, end, oid = get_info(idx[0], house[0]) 
            v_m = v_var[0][0]
            f_m = f_var[0][0][begin:end]
            vt_m = vt_var[0][0][begin:end]
            nr.save_obj('/path/to/data/house/' + house[0] + '/attack_' + str(int(idx[0])) + '_' + str(oid) + '.obj', v_m, f_m, vt_m)
        if all_envs_loaded == False:
            eval_loader.dataset._load_envs()
            if len(eval_loader.dataset.pruned_env_set) == 0:
                done = True
        else:
            done = True
          
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
    parser.add_argument('-max_threads_per_gpu', default=1, type=int)

    # checkpointing
    parser.add_argument('-checkpoint_path', default='/home/neuEmbodiedQA/training/checkpoints/vqa/08_15_23:02_ques-image/epoch_990_accuracy_0.6485.pt')
    parser.add_argument('-checkpoint_dir', default='checkpoints/vqa/')
    parser.add_argument('-log_dir', default='logs/vqa/')
    parser.add_argument('-log', default=False, action='store_true')
    parser.add_argument('-cache', default=False, action='store_true')

    #parser.add_argument(''.)
    args = parser.parse_args()

    args.time_id = time.strftime("%m_%d_%H:%M")

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
    attack_fgsm(0, args, shared_model, 0)

    
# CUDA_VISIBLE_DEVICES=0,1 python attack.py










