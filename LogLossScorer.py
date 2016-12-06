from LossScorer import *

class LogLossScorer(LossScorer):
    """A log loss scorer class"""
    def __init__(self, train_file, n_training_samples, lm_weight, reg, give_tag):
        LossScorer.__init__(self,train_file, n_training_samples, lm_weight, reg, give_tag)
    
    def local_loss(self, training_sample_features, gold_label, weights):
        vg = self.get_vec(training_sample_features, self.pos_map, self.n_features, weights.size(), gold_label)
        
        v = weights.dot(v)
        
        
        
    def iterate(self, weights, index):
        """Iterate a single training sample"""
        return
    


    def gen_local_pos_features(self, local_sent_features, pos_map, n_pos_tags, gold):
        features = {}

        partial_sum = 0
        for w in local_sent_features:
            p = pos_map[w]
            # print w,p
    
            l =len(local_sent_features[w])
            if (l > 1):
                local_pos_features[w] = local_sent_features[w]
            else:
                partial_sum += exp(lm_weight*local_pos_features[w][0])
    
        return [features,partial_sum]