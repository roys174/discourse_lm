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

    # Read feature file
    [n_features, n_pos_tags, pos_map, feature_sets,gold_vec] = read_training_file(ifile)
    if (feature_sets == None):
        return -2
    
    total_n_features = n_features+n_pos_tags
    print(str(total_n_features)+", "+str(n_pos_tags)+", "+str(len(feature_sets))+", "+str(len(pos_map)))
    
    # Initializing weights as random number between (-1,1)
    weights = [random.uniform(-1,1) for i in range(total_n_features)]
    weights = [0 for i in range(total_n_features)]
    # Weight for LM value is fixed to 1. 
    weights[n_pos_tags] = 1
        
    # Iterating data.
    for i in range(n_iter):
        # A single iteration
        [loss, n_true, weights] = run_iteration(feature_sets,pos_map, n_pos_tags, n_features, weights, gold_vec, reg)
        print str(i)+": "+str(loss)+". #true="+str(n_true)
    
    # Saving result
    with FileTools.openWriteFile(ofile) as ofh:
        for i in range(total_n_features):
            ofh.write(str(weights[i])+"\n")
    
    return 0

# A single iteration of the entire dataset.
def run_iteration(training_set,pos_map, n_pos_tags, n_features, weights, gold_vec, reg):
    n_sent = 0
    for training_sample_features in training_set:
        # Iterating one training sample
        local_weights_delta = iterate_sample(training_sample_features, pos_map, n_pos_tags, n_features, weights, gold_vec[n_sent], reg)

        n_sent = n_sent + 1

        if (len(local_weights_delta) > 0):
            # Do not update language model feature.
            local_weights_delta[n_pos_tags] = 0
            weights = np.add(weights, local_weights_delta)
        
            
#    weights = np.add(weights,weights_delta)
#    print "de is "+str(weights_delta)
#    print "n is "+str(weights)
    
    [loss,n_true] = compute_loss(n_pos_tags, n_features, training_set, pos_map, gold_vec, weights,reg)
    return [loss,n_true, weights]
    

def compute_loss(n_pos_tags, n_features, training_samples, pos_map, gold_vec, weights,reg):
    n_sent = 0
    v = [0 for x in range(n_pos_tags+n_features)]
    n_true = 0
    for training_sample_features in training_samples:
        [local_l,is_correct] = local_loss(training_sample_features, pos_map, n_pos_tags, n_features, weights, gold_vec[n_sent], reg)
        v = np.add(v, local_l)
        n_sent = n_sent + 1
        n_true  = n_true + is_correct
    
    
    return [np.dot(weights,v) - reg*np.linalg.norm(weights), n_true]
    

def local_loss(training_sample_features, pos_map, n_pos_tags, n_features, weights, gold, reg):
    max_val = get_max(training_sample_features, n_pos_tags, pos_map, weights)
    vg = get_vec(training_sample_features, pos_map, n_pos_tags, n_features, gold)
    vam = get_vec(training_sample_features, pos_map, n_pos_tags, n_features, max_val)
    
    return [np.subtract(vg,vam),gold==max_val]
    

def iterate_sample(training_sample_features, pos_map, n_pos_tags, n_features, weights, gold, reg):
    # First, get predicted word.
    max_val = get_max(training_sample_features, n_pos_tags, pos_map, weights)
    
    # print max_val+", "+gold
    
    # If predicted value does not match gold value, update weights.
    if (max_val != gold):
        # update w(t+1) = w(t) + vec(gold) - vec(predicted) - 2*reg*w(t)
        vg = get_vec(training_sample_features, pos_map,n_pos_tags, n_features, gold)
        vam = get_vec(training_sample_features, pos_map, n_pos_tags, n_features, max_val)
        reg_val = np.multiply(weights,2*reg)
        
        return np.subtract(vg,np.add(vam,reg_val))
    else:
        return []

# @todo: implement method
def get_vec (feature_sets, pos_map, n_pos_tags, n_features, w):
    vec = [0 for i in range(n_pos_tags+n_features)]

    vec[pos_map[w]] = 1
    
    # print w+" is w. pos is "+str(pos_map[w])
    for f in feature_sets[pos_map[w]][w]:
        # print "f is "+str(f)
        vec[n_pos_tags+f] = feature_sets[pos_map[w]][w][f]
    
    #print vec
    return vec

# Maximum value
def get_max(feature_sets, n_pos_tags, pos_map, weights):
    max_vals = []
    argmax_vals = []

    # printfeature_sets
    for i in range(n_pos_tags):
        # printi
        [m,am] = get_local_max(feature_sets[i], weights, n_pos_tags)
        max_vals.append(m+weights[i])
        
        argmax_vals.append(am)

    return argmax_vals[np.argmax(max_vals)]
            
# Candidate for a given PoS tag.          
def get_local_max(feature_set, weights, n_pos_tags):
    max_val = -10000
    arg_max = ''
    # printfeature_set
    for w in feature_set:
        # print"'"+w+"'"

        v = 0
        for f in feature_set[w]:
            v = v + weights[n_pos_tags+f]*feature_set[w][f]
        
        if v > max_val:
            max_val = v
            arg_max = w
        
    #print str(max_val)+", "+str(arg_max)
    return [max_val,arg_max]
    

def read_training_file(train_file):
    pos_map = {}
    gold_vec = []

    with FileTools.openReadFile(train_file) as ifile:
        info = ifile.readline().rstrip()
            
        e = info.split(" ")
        n_pos_tags = int(e[0])
        n_features = int(e[1])
        features = []
        
        # ignore next line
        line = ifile.readline()
        s = {}
        # print n_pos_tags,n_features
        if (read_features(s, 1, ifile) == 0):
            return [None,None]
            
        pos_map = dict([(w,int(s[w][0])-1) for w in s])
        
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
            
            
            local_sent_features = {}
            
            if (read_features(local_sent_features, n_features, ifile) == 0):
                return [None,None]
            
            if (gold in pos_map):
                gold_vec.append(gold)
                
                local_pos_features = [{} for i in range(n_pos_tags)]
                
                features.append(gen_local_pos_features(local_sent_features, pos_map, n_pos_tags, gold))
                                
            line = ifile.readline()
            # if n_sent == 30:
                # break
            if (n_sent %10 == 0):
                print(str(n_sent))#, end='\r')
#                sys.stdout.flush()
                # break
                if (n_sent % 10000 == 0):
                    break
        #print "\n"+str(n_sent)+" "+line
    
    return [n_features,n_pos_tags, pos_map, features,gold_vec]

def gen_local_pos_features(local_sent_features, pos_map, n_pos_tags, gold):
    local_pos_features = [{} for i in range(n_pos_tags)]
    
    max_pos_values = [-10000 for i in range(n_pos_tags)]
    argmax_pos_values = ['' for i in range(n_pos_tags)]
    
    for w in local_sent_features:
        p = pos_map[w]
        # print w,p
        
        l =len(local_sent_features[w])
        if (l > 1):
            local_pos_features[p][w] = local_sent_features[w]
        else:
            if local_sent_features[w][0] > max_pos_values[p]:
                max_pos_values[p] = local_sent_features[w][0]
                argmax_pos_values[p] = w
        
    for i in range(n_pos_tags):
        local_pos_features[i][argmax_pos_values[i]] = {0: max_pos_values[i]}
    
#    print "Gold is "+gold+" ("+str(pos_map[gold])+")"
    if gold not in local_pos_features[pos_map[gold]]:
        local_pos_features[pos_map[gold]][gold] = {0: local_sent_features[gold][0]}
        
    return local_pos_features
    
    
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
            
            v=float(e[1])
            
            features[w][i] = v

                
    #    print "Read "+str(len(features[0]))+" features"
    return 1

sys.exit(main())
    