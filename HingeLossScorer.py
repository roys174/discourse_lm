from LossScorer import *

class HingeLossScorer(LossScorer):
    """A hinge loss scorer class"""
    def __init__(self, n_pos_tags, n_features, pos_map, reg, give_tag):
        LossScorer.__init__(self,n_pos_tags, n_features, pos_map, reg, give_tag)
    
    def local_loss(self, training_sample_features, gold_label, weights):
        tag = None
        if self.give_tag:
            tag=pos_map[gold]
        
        max_val = get_max(training_sample_features, self.n_pos_tags, self.n_features, self.pos_map, weights, tag)
        
        vg = get_vec(training_sample_features, self.pos_map, self.n_features, len(weights), gold_label)
        vam = get_vec(training_sample_features, self.pos_map, self.n_features, len(weights), max_val)

    #    print gold_label+": "+str(training_sample_features[pos_map[gold_label]][gold_label])+", "+max_val+": "+str(training_sample_features[pos_map[max_val]][max_val])+"# "+str(gold_label==max_val)

        return [np.subtract(vg,vam),gold_label==max_val]
    
    
    def iterate(self, training_sample_features, weights, gold):
        """Iterate a single training sample"""
        tag = None
        if self.give_tag:
            tag=self.pos_map[gold]

        # print tag
        # First, get predicted word.
        max_val = get_max(training_sample_features, self.n_pos_tags, self.n_features, self.pos_map, weights, tag)

        # print gold+", "+max_val+": "+str(gold==max_val)
        # print gold+": "+str(training_sample_features[pos_map[gold]][gold])+", "+max_val+": "+str(training_sample_features[pos_map[max_val]][max_val])+"# "+str(gold==max_val)

        # If predicted value does not match gold value, update weights.
        if (max_val != gold):
            # update w(t+1) = w(t) + vec(gold) - vec(predicted) - 2*reg*w(t)
            vg = get_vec(training_sample_features, self.pos_map, self.n_features, len(weights), gold)
            vam = get_vec(training_sample_features, self.pos_map, self.n_features, len(weights), max_val)
            reg_val = np.multiply(weights,2*self.reg)
        #        print "g",training_sample_features[pos_map[gold]][gold],pos_map[gold],1+pos_map[gold]*n_features
        #       print "am",training_sample_features[pos_map[max_val]][max_val],pos_map[max_val],1+pos_map[max_val]*n_features
            # print "Updating!"
            return np.subtract(vg,np.add(vam,reg_val))
        else:
            return []
    
