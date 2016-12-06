from LossScorer import *

class HingeLossScorer(LossScorer):
    """A hinge loss scorer class"""
    def __init__(self, train_file, n_training_samples, lm_weight, reg, give_tag):
        LossScorer.__init__(self,train_file, n_training_samples, lm_weight, reg, give_tag)
    
    def local_loss(self, training_sample_features, gold_label, weights):
        tag = None
        if self.give_tag:
            tag=pos_map[gold]
        
        max_val = self.get_max(training_sample_features, self.n_pos_tags, self.pos_map, weights, tag)
        
        vg = self.get_vec(training_sample_features, self.pos_map, self.n_features, weights.size(), gold_label)
        vam = self.get_vec(training_sample_features, self.pos_map, self.n_features, weights.size(), max_val)

    #    print gold_label+": "+str(training_sample_features[pos_map[gold_label]][gold_label])+", "+max_val+": "+str(training_sample_features[pos_map[max_val]][max_val])+"# "+str(gold_label==max_val)

        return [vg - vam,gold_label==max_val]

    def finish_v(self, v, weights):
        """Run another operation on the vector before regulariztion"""
        return weights.dot(v)

    
    def iterate(self, weights, index):
        """Iterate a single training sample"""
        tag = None
        gold=self.gold_vec[index]
        training_sample_features = self.features[index]
        
        if self.give_tag:
            tag=self.pos_map[gold]

        # print tag
        # First, get predicted word.
        max_val = self.get_max(training_sample_features, self.n_pos_tags, self.pos_map, weights, tag)

        # print gold+", "+max_val+": "+str(gold==max_val)
        # print gold+": "+str(training_sample_features[pos_map[gold]][gold])+", "+max_val+": "+str(training_sample_features[pos_map[max_val]][max_val])+"# "+str(gold==max_val)

        # If predicted value does not match gold value, update weights.
        if (max_val != gold):
            # update w(t+1) = w(t) + vec(gold) - vec(predicted) - 2*reg*w(t)
            vg = self.get_vec(training_sample_features, self.pos_map, self.n_features, weights.size(), gold)
            vam = self.get_vec(training_sample_features, self.pos_map, self.n_features, weights.size(), max_val)
#            reg_val = np.multiply(weights.get(),2*self.reg)
            reg_val = weights*(2*self.reg)
        #        print "g",training_sample_features[pos_map[gold]][gold],pos_map[gold],1+pos_map[gold]*n_features
        #       print "am",training_sample_features[pos_map[max_val]][max_val],pos_map[max_val],1+pos_map[max_val]*n_features
            # print "Updating!"
            return vg-(vam+reg_val)
            # return np.subtract(vg,np.add(vam,reg_val))
        else:
            return FeatureVector(0,0,2,0)
    

    def gen_local_pos_features(self, local_sent_features, pos_map, n_pos_tags, gold):
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
    
        return [local_pos_features,0]