# Combine all changed .obj file (each representing an attacked object) to one, thus can be rendered as a house 


import os
import json
import csv

def find_index(name, house):
    rf_o = open('/path/to/data/house/' + house + '/backup/house.obj')
    idx = 0
    s = 0
    for line in rf_o:
        line_s = line.split(' ')[0]
        if(line_s == 'g' and s==1):#FIRST LAST
            if(line.split('#')[0] == 'g Object'):
                end = idx
                rf_o.close()
                return begin, end
        if(line == 'g Object#0_' + str(name) +'\n'):# FIRST LAST
            begin = idx
            s = 1
        if(line_s == 'f'):
            idx = idx + 1
    rf_o.close()
    begin = 1000
    end = 2000
    return begin, end



def handle(path):
    base_path = path.split('attack')[0]
    house_id = path.split('/')[-2]
    que_id = path.split('/')[-1].split('_')[-2]
    obj_id = path.split('/')[-1].split('_')[-1].split('.')[-2]
    rf_o = open(path)
    wf_a = open(base_path + 'att1.obj','w')
    for line in rf_o:
        line_s = line.split(' ')[0]
        if(line_s != 'v'):
            wf_a.write(line)

    rf_o.close()
    wf_a.close()

    num = 0
    rf_o = open(base_path + 'house_' + que_id + '.obj')
    for line in rf_o:
        line_s = line.split(' ')[0]
        if(line_s == 'vt'):
            num += 1
    rf_o.close()

    rf_o = open(base_path + 'att1.obj')
    wf_a = open(base_path + 'att2.obj','w')

    for line in rf_o:
        line_s = line.split(' ')
        ln0 = line_s[0]
        if(ln0 != 'f'):
            wf_a.write(line)
        else:
            ls1 = line_s[1]
            ls2 = line_s[2]
            ls3 = line_s[3][:-1]
            ls1_n = int(ls1.split('/')[1]) + num
            ls1 = ls1.split('/')[0] + '/' + str(ls1_n) + '/' + ls1.split('/')[0]
            ls2_n = int(ls2.split('/')[1]) + num
            ls2 = ls2.split('/')[0] + '/' + str(ls2_n) + '/' + ls2.split('/')[0]
            ls3_n = int(ls3.split('/')[1]) + num
            ls3 = ls3.split('/')[0] + '/' + str(ls3_n) + '/' + ls3.split('/')[0]
            ln = 'f ' + ls1 + ' ' + ls2 + ' ' + ls3 + '\n'
            wf_a.write(ln)

    rf_o.close()
    wf_a.close()

    os.system('rm ' + base_path + 'att1.obj')

    rf_o = open(base_path + 'house_' + que_id + '.obj')
    rf_o1 = open(base_path + 'att2.obj')
    wf_a = open(base_path + 'att3.obj','w')

    for line in rf_o:
        if(line == 'mtllib house.mtl\n'):
            wf_a.write('mtllib ' + 'house.mtl\n') 
            continue
        wf_a.write(line)

    js = 0
    for line in rf_o1:
        js += 1
        if(js <= 6):
            continue
        wf_a.write(line)

    rf_o.close()
    rf_o1.close()
    wf_a.close()

    os.system('rm ' + base_path + 'att2.obj')


    rf_o = open(base_path + 'att3.obj')
    wf_a = open(base_path + 'house_' + que_id + '.obj','w')

    begin, end = find_index(obj_id, house_id)
    idx = 0 

    for line in rf_o:
        line_s = line.split(' ')[0]
        if(line_s == 'f'):
            idx = idx + 1
            if(idx>=begin + 1 and idx<=end):
                continue
        wf_a.write(line)

    rf_o.close()
    wf_a.close()
    os.system('rm ' + base_path + 'att3.obj')


    # mtl
    rf_o = open(base_path + 'house.mtl')
    rf_o1 = open(base_path + 'attack_' + que_id + '_' + obj_id + '.mtl')
    wf_a = open(base_path + 'house1.mtl','w')

    for line in rf_o:
        wf_a.write(line)

    wf_a.write('\n')

    for line in rf_o1:
        wf_a.write(line) 
        break
    wf_a.write('Ka 0.64 0.64 0.64\n')
    wf_a.write('Kd 0.64 0.64 0.64\n')
    for line in rf_o1:
        wf_a.write(line) 
        break


    rf_o.close()
    rf_o1.close()
    wf_a.close()
    os.system('cp ' + base_path + 'house1.mtl ' + base_path + 'house.mtl')


if __name__ == '__main__':
    done = [your house]
    with open('data.json', 'r') as f:
        res = json.load(f)     
    for num in range(1):
        data = res[done[num]]
        for cvpr in range(len(data)):
            qid = data[cvpr]
            os.system('cp /path/to/data/house/' + done[num] + '/house.obj /path/to/data/house/' + done[num] + '/house_' + qid + '.obj')

    path = []
    for idx in range(len(done)):
        for root, dirs, files in os.walk('/path/to/data/house/' + done[idx]):
            for f in files:
                path.append(root+'/'+f)
        break
    for idx in range(len(path)):
        if (path[idx].split('/')[-1].split('_')[0] == 'attack' and path[idx].split('.')[-1] == 'obj'):
            handle(path[idx])


