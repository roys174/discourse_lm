import abc
from max_ent_funcs import *


class LossScorer:
    """A general scorer class"""
    def __init__(self, n_pos_tags, n_features, pos_map, reg, give_tag):
        self.n_pos_tags = n_pos_tags
        self.n_features = n_features
        self.pos_map = pos_map
        self.reg = reg
        self.give_tag = give_tag
    
    def compute_loss(self, training_samples, gold_vec, weights):
        n_sent = 0
        v = [0 for x in range(len(weights))]
        n_true = 0
        for training_sample_features in training_samples:
            # print n_sent," #######"
            [local_l,is_correct] = self.local_loss(training_sample_features, gold_vec[n_sent], weights)
            v = np.add(v, local_l)
            n_sent = n_sent + 1
            n_true  = n_true + is_correct
    
        return [np.dot(weights,v) - self.reg*np.linalg.norm(weights), n_true]
    
    @abc.abstractmethod
    def local_loss(self, training_sample_features, gold_label, weights):
        """Compute loss"""
        return
        
    @abc.abstractmethod
    def iterate(self, training_sample_features, weights, gold):
        """Iterate a single training sample"""
        return