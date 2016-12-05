from LossScorer import *

class LogLossScorer(LossScorer):
    """A log loss scorer class"""
    def __init__(self, n_pos_tags, n_features, pos_map, reg, give_tag):
        LossScorer.__init__(self,n_pos_tags, n_features, pos_map, reg, give_tag)
    
    def local_loss(self, training_sample_features, gold_label, weights):
        return
        
    def iterate(self, training_sample_features, weights, gold):
        """Iterate a single training sample"""
        return
    
