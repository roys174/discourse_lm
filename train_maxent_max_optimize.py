#!/usr/bin/python 

from LogLossScorer import *
from HingeLossScorer import *
import sys

from optparse import OptionParser,OptionValueError

def main():
    options = usage()
       
    ifile = options.train_file
    ofile = options.out_file
    n_iter = int(options.n_iter)
    reg = float(options.reg)
    give_tag = bool(options.give_tag)
    step_size_ratio = float(options.step_size_ratio)
    lm_weight = int(options.lm_weight)
    n_training_samples = int(options.n_training_samples)
    scorer_index = int(options.scorer_index)
    
    print give_tag,n_training_samples
        
    if (scorer_index == 1):
        scorer = LogLossScorer(ifile, n_training_samples, lm_weight, reg, give_tag)
    elif (scorer_index == 2):
        scorer = HingeLossScorer(ifile, n_training_samples, lm_weight, reg, give_tag)
    else:
        print "scorer must be either 1 or 2"
        return -1
        
    # # Read feature file
    # [n_features, n_pos_tags, pos_map, feature_sets,gold_vec] = read_training_file(ifile, n_training_sample)
    # if (feature_sets == None):
    #     return -2
    #
    total_n_features = (scorer.n_features)*scorer.n_pos_tags+1
    # print(str(total_n_features)+", "+str(n_pos_tags)+", "+str(len(feature_sets))+", "+str(len(pos_map)))
    #
    weights = FeatureVector(total_n_features, lm_weight, 1, scorer.n_features)
    
    step_size = 1

    print scorer_index
    
        
    # Iterating data.
    for i in range(n_iter):
        # print "#### ", i
        # A single iteration
        [loss, n_true, weights] = run_iteration(scorer, weights, reg,give_tag, step_size)
        print str(i)+": "+str(loss)+". #true="+str(1.*n_true/len(scorer.features))+"="+str(n_true)+"/"+str(len(scorer.features))+". Step size="+str(step_size)
        step_size = step_size*step_size_ratio
    
    # Saving result
    with FileTools.openWriteFile(ofile) as ofh:
        for i in range(total_n_features):
            ofh.write(str(weights[i])+"\n")
    
    return 0

# A single iteration of the entire dataset.
def run_iteration(scorer, weights, reg,give_tag, step_size):
    n_sent = 0
    indices=range(len(scorer.features))
    np.random.shuffle(indices)
    for i in indices:
        # print i
        # Iterating one training sample
        local_weights_delta = scorer.iterate(weights,i)

        # print n_sent
        if (local_weights_delta.size() > 0):
#            print "Updating 2!"
            # Do not update language model feature.
            local_weights_delta[0] = 0

            # delta_dict = {}
        #
        #     for i in range(len(local_weights_delta)):
        #         if (local_weights_delta[i] != 0):
        #             delta_dict[i] = local_weights_delta[i]
        #
        #     print delta_dict
            # a = 1+pos_map[gold_vec[i]]*n_features
            # b = 1+(pos_map[gold_vec[i]]+1)*n_features
            # print a,b,gold_vec[i],pos_map[gold_vec[i]]
            # print str(weights[a:b])
#            print weights
            weights += (local_weights_delta * step_size)
#            weights.set(np.add(weights.get(), np.multiply(local_weights_delta, step_size)))
            # print str(weights[a:b])
 #           print weights
        
        n_sent = n_sent + 1
            
#    weights = np.add(weights,weights_delta)
#    print "de is "+str(weights_delta)
#    print "n is "+str(weights)
    
    [loss,n_true] = scorer.compute_loss(weights)
    return [loss,n_true, weights]
    

def usage():
        parser = OptionParser()
        n_iter = 100
        reg = 0.1
        step_size_ratio = 0.9
        lm_weight = 1
        n_training_samples = 100
        scorer_index=2
    
        parser.add_option("-t", dest="train_file",
                        help="training file", metavar="FILE")
        parser.add_option("-o", dest="out_file",
                        help="Output file", metavar="FILE")
        parser.add_option("-n", metavar="INTEGER",
                                dest="n_iter", help="Number of iterations",
                                default=n_iter)
        parser.add_option("-r", metavar="FLOAT",
                                dest="reg", 
                                help="Regularization parameter",
                                default=reg)
        parser.add_option("-s", metavar="FLOAT",
                                dest="step_size_ratio", 
                                help="Step size ratio",
                                default=step_size_ratio)
        parser.add_option("-c", metavar="INTEGER",
                                dest="scorer_index", 
                                help="Score type (1 for log loss, 2 for hinge (default))",
                                default=scorer_index)
        parser.add_option("-g", dest="give_tag", default=False, action="store_true",
                                help="Use gold part-of-speech tags")
        parser.add_option("-l", metavar="INTEGER",
                                dest="lm_weight", 
                                help="Weight for LM feature",
                                default=lm_weight)
        parser.add_option("-i", metavar="INTEGER",
                                dest="n_training_samples", 
                                help="Number of training samples to train on",
                                default=n_training_samples)
        
        
        (options, args) = parser.parse_args()
        
        if (options.train_file == None or options.out_file == None):
                raise OptionValueError("both file arguments are required")
                
        return options


sys.exit(main())
    