#!/usr/bin/python 

import sys
import FileTools
import random
import numpy as np


def main():
    argc = len(sys.argv)    
    
    n_iter = 100
    reg = 0.1
    
    if (argc < 3):
        print("Usage: "+sys.argv[0]+" <training file> <out file> <n_iter = "+str(n_iter)+"> <reg = "+str(reg)+">")
        return -1
    elif (argc > 3):
        n_iter = int(sys.argv[3])
        if (argc > 4):
            reg = float(sys.argv[4])

    ifile = sys.argv[1]
    ofile = sys.argv[2]
    
    [n_local, n_global, global_features, local_features,gold_vec] = read_training_file(ifile)
    if (local_features == None):
        return -2
    
    n_features = n_local+n_global
    print(str(n_local)+", "+str(n_global)+", "+str(len(global_features))+", "+str(len(local_features))+", "+str(len(local_features[0])))
    
    weights = [random.uniform(-1,1) for i in range(n_features)]
    #weights = [1 for i in range(n_features)]
    print len(global_features.values())
    
    for i in range(n_iter):
        [loss, weights] = run_iteration(n_features,local_features,global_features,n_global, n_local, weights, gold_vec, reg)
        print str(i)+": "+str(loss)
    
    with FileTools.openWriteFile(ofile) as ofh:
        for i in range(n_features):
            ofh.write(str(weights[i])+"\n")
    
    return 0


def run_iteration(n_features, training_samples, global_features,n_global, n_local, weights, gold_vec, reg):
    n_sent = 0
    for s in training_samples:
        local_weights_delta = iterate_sample(s, global_features, n_global, n_local, weights, gold_vec[n_sent], reg)
        n_sent = n_sent + 1

        if (len(local_weights_delta) > 0):
            weights = np.add(weights, local_weights_delta)
        
            
#    weights = np.add(weights,weights_delta)
#    print "de is "+str(weights_delta)
#    print "n is "+str(weights)
    
    loss = compute_loss(n_global, n_local, training_samples, global_features, gold_vec, weights,reg)
    return [loss,weights]
    

def compute_loss(n_global, n_local, training_samples, global_features, gold_vec, weights,reg):
    n_sent = 0
    v = [0 for x in range(n_global+n_local)]
    
    for s in training_samples:
        v = np.add(v, local_loss(s, global_features, n_global, n_local, weights, gold_vec[n_sent], reg))
        n_sent = n_sent + 1
    
    
    return np.dot(weights,v) - reg*np.linalg.norm(weights)
    

def local_loss(local_features, global_features, n_global, n_local, weights, gold, reg):
    max_val = get_max(global_features, local_features, n_global, weights)
    vg = get_vec(global_features, local_features, n_global, n_local, gold)
    vam = get_vec(global_features, local_features, n_global, n_local, max_val)
    
    return np.subtract(vg,vam)
    

def iterate_sample(local_features, global_features, n_global, n_local, weights, gold, reg):
    if (gold not in global_features):
        #print gold+" not found in vocab. Skipping sample"
        return []
        
    max_val = get_max(global_features, local_features, n_global, weights)
    #print max_val+", "+gold
    if (max_val != gold):
        # update!
        vg = get_vec(global_features, local_features, n_global, n_local, gold)
        vam = get_vec(global_features, local_features, n_global, n_local, max_val)
        reg_val = np.multiply(weights,reg)
        
        return np.subtract(vg,np.add(vam,reg_val))
    else:
        return []

# @todo: implement method
def get_vec (global_features, local_features, n_global, n_local, w):
    vec = [0 for i in range(n_global+n_local)]
    #print w+", "+str(global_features[w])
    vec[global_features[w]] = 1
    
    for f in local_features[w]:
        vec[n_global+f] = local_features[w][f]
    
    #print vec
    return vec

# @todo: implement method
def get_max(global_features, local_features, n_global, weights):
    max_val = -10000
    arg_max = ''
    for w in local_features:
        # print w
        # print global_features[w]
        v = weights[global_features[w]]
        for f in local_features[w]:
            v = v + weights[n_global+f]*local_features[w][f]
        
        if v > max_val:
            max_val = v
            arg_max = w
        
    #print str(max_val)+", "+str(arg_max)
    return arg_max

def read_training_file(train_file):
    global_features = {}
    local_features = []
    gold_vec = []

    with FileTools.openReadFile(train_file) as ifile:
        info = ifile.readline().rstrip()
            
        e = info.split(" ")
        n_global = int(e[0])
        n_local = int(e[1])
        
        # ignore next line
        line = ifile.readline()
        # print n_global,n_local
        if (read_features(global_features, 1, ifile) == 0):
            return [None,None]
        
        global_features = dict([(x,int(global_features[x][0])-1) for x in global_features])
        
        line = ifile.readline()
    
        if (len(line) == 0):
            return None
    
        
        n_sent = 0

        while (len(line) > 0):
            # print str(n_sent)+": "+line
            if (line[0] != '#'):
                print "Expecting comment in line "+line
                return None
        
            e = line.rstrip().split(" ")
            gold = e[-1]
            
            n_sent = n_sent + 1
            
            if (n_sent %10 == 0):
                print(str(n_sent))#, end='\r')
#                sys.stdout.flush()
                if (n_sent % 100 == 0):
                    break
            
            local_sent_features = {}
            if (read_features(local_sent_features, n_local, ifile) == 0):
                return [None,None]
            
            if (gold in global_features):
                gold_vec.append(gold)
                local_features.append(local_sent_features)
                
            line = ifile.readline()
        #print "\n"+str(n_sent)+" "+line
    
    return [n_local,n_global, global_features, local_features,gold_vec]

    
def read_features(features, n_features, ifile):
    # print n_features
    for i in range(n_features):
        line = ifile.readline()
                
        if (line == None):
            print "Not enough lines if "+ifile
            return 0
        
        line = line.rstrip()
        
        #print line
        for x in line.split(" "):
            e = x.split(":")
            w = e[0]
            
            if w not in features:
                features[w] = {}
            
            
            features[w][i] = float(e[1])
                
    #    print "Read "+str(len(features[0]))+" features"
    return 1

sys.exit(main())
    