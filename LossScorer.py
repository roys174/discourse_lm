import abc
from max_ent_funcs import *
import FileTools
from FeatureVector import *
import sys

class LossScorer:
    """A general scorer class"""

    def __init__(self, train_file, n_training_samples, lm_weight, reg, give_tag):
        self.reg = reg
        self.give_tag = give_tag
        self.lm_weight = lm_weight
        
        self.read_training_file(train_file, n_training_samples)
    
    def compute_loss(self, weights):
        n_sent = 0
        v = FeatureVector(weights.size(), 0, 2, weights.n_features)
#        v = [0 for x in range(weights.size())]
        n_true = 0
        for training_sample_features in self.features:
            # print n_sent," #######"
            [local_l,is_correct] = self.local_loss(training_sample_features, self.gold_vec[n_sent], weights)
            v += local_l
            n_sent = n_sent + 1
            n_true  = n_true + is_correct
    
        
        v = self.finish_v(v,weights)
    
        return [v - self.reg*weights.l2(), n_true]
    
    
    def finish_v(self, v,weights):
        """Run another operation on the vector before regulariztion"""
        return v
    
    @abc.abstractmethod
    def local_loss(self, training_sample_features, gold_label, weights):
        """Compute loss"""
        return
        
    @abc.abstractmethod
    def iterate(self, weights, index):
        """Iterate a single training sample"""
        return
      
      
    def read_training_file(self, train_file, n_training_samples=-1):
        pos_map = {}
        gold_vec = []

        with FileTools.openReadFile(train_file) as ifile:
            info = ifile.readline().rstrip()
            
            e = info.split(" ")
            n_pos_tags = int(e[0])
            n_features = int(e[1])
            features = []
            sums = []
        
            # ignore next line
            line = ifile.readline()
            s = {}
            # print n_pos_tags,n_features
            if (self.read_features(s, 1, ifile) == 0):
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
            
                if (self.read_features(local_sent_features, n_features, ifile) == 0):
                    return [None,None]
            
                if (gold in pos_map):
                    gold_vec.append(gold)
                
                    arr = self.gen_local_pos_features(local_sent_features, pos_map, n_pos_tags, gold)
                
                    
                    features.append(arr[0])
                    sums.append(arr[1])
                                
                line = ifile.readline()
                # break
                # if n_sent == 3:
                #     break
            
                if (n_training_samples > 0 and n_sent == n_training_samples):
                    break
                
                if (n_sent %10 == 0):
                    print str(n_sent)+"    \r",
                    sys.stdout.flush()
               
            #print "\n"+str(n_sent)+" "+line
    
        self.n_features = n_features
        self.n_pos_tags = n_pos_tags
        self.pos_map = pos_map
        self.gold_vec = gold_vec
        self.features = features
        self.sums = sums
    
        return

            

    def read_features(self, features, n_features, ifile):
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
                if (len(e) != 2):
    #                print "Bad line '"+line+"'"
                    continue
            
                w = e[0]
        
                if w not in features:
                    features[w] = {}
        
                v=float(e[1])
        
                features[w][i] = v

            
        #    print "Read "+str(len(features[0]))+" features"
        return 1

    @abc.abstractmethod     
    def gen_local_pos_features(self, local_sent_features, pos_map, n_pos_tags, gold):
        return


    # Get feature vector for word
    def get_vec(self, feature_sets, pos_map, n_features, n_total_features,w):
        vec = FeatureVector(n_total_features, feature_sets[pos_map[w]][w][0], 2,n_features)

        # PoS indicator feature
        vec.set_feature_value(1, pos_map[w])

        # print w+" is w. pos is "+str(pos_map[w])
        for f in feature_sets[pos_map[w]][w]:
            if (int(f) == 0):
                continue
        
            # print "f is "+str(f)+" ("+str(pos_map[w]*n_features+f)+")"
            vec.set_feature_value(feature_sets[pos_map[w]][w][f], pos_map[w],f)

        #print vec
        return vec

    # Maximum value
    def get_max(self, feature_sets, n_pos_tags, pos_map, weights, tag=None):
        max_vals = []
        argmax_vals = []

        if (tag==None):
            # print feature_sets
            for i in range(n_pos_tags):
                # print i
                # print feature_sets[i]
                [m,am] = self.get_local_max(i, feature_sets[i], weights)

                # Add pos indicator feature.
                max_vals.append(m+weights.get_feature_value(i))

                argmax_vals.append(am)

            return argmax_vals[np.argmax(max_vals)]
        else:
            [m,am] = self.get_local_max(tag, feature_sets[tag], weights)
            return am
        
    # Candidate for a given PoS tag.          
    def get_local_max(self, i,feature_set, weights):
        max_val = -10000
        arg_max = ''
        # printfeature_set
        for w in feature_set:
            # print"'"+w+"'"

            v = feature_set[w][0]
            for f in feature_set[w]:
                if (f==0):
                    continue
            
                v = v + weights.get_feature_value(i,f)*feature_set[w][f]
    
            if v > max_val:
                max_val = v
                arg_max = w
    
        #print str(max_val)+", "+str(arg_max)
        return [max_val,arg_max]

    