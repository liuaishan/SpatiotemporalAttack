# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import subprocess
import shlex
import os
import multiprocessing

parser = argparse.ArgumentParser(
    description='Create obj+mtl files for the houses in the dataset.')
parser.add_argument('-eqa_path', help='/path/to/eqa.json', required=True)
parser.add_argument(
    '-dataset_toolbox_path', help='/path/to/toolbox', required=True)
parser.add_argument(
    '-dataset_data_path', help='/path/to/dataset/data_root', required=True)
parser.add_argument(
    '-num_processes',
    help='number of threads to use',
    default=multiprocessing.cpu_count())
args = parser.parse_args()

eqa_data = json.load(open(args.eqa_path, 'r'))
houses = list(eqa_data['questions'].keys())
start_dir = os.getcwd()


def extract_threaded(house):
    os.chdir(os.path.join(args.dataset_data_path, 'house', house))
    subprocess.call(
        shlex.split('sudo %s house.json house.obj' % (os.path.join(
            args.dataset_toolbox_path, 'gaps', 'bin', 'x86_64', 'scn2scn'), )))
    print('extracted', house)


pool = multiprocessing.Pool(args.num_processes)
pool.map(extract_threaded, houses)