import os
import random

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

def readfile(filename):
    '''read file'''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir, train_data, size=None):
        """See base class."""
        train_file = self._read_tsv(os.path.join(data_dir, train_data))
        return_example = self._create_examples(train_file, "train")
        if size:
          select_idx = np.random.choice(range(len(return_example)), size=size, replace=False)
          return_example = list(np.array(return_example)[select_idx])
        return return_example

    def get_dev_examples(self, data_dir, dev_data):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, dev_data)), "dev")
    
    def get_test_examples(self, data_dir, test_data):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, test_data)), "test")
    def get_labels(self):
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
        #return ["O", "B-chem", "I-chem", "[CLS]", "[SEP]"]
        #return ["O", "B-disease", "I-disease", "[CLS]", "[SEP]"]
    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, logger, start_end=None, ori_len=None, predicted_label=None, iterative_len=None, do_sampling=True):
    """Loads a data file into a list of `InputBatch`s."""
    
    
    if do_sampling:
        pass
    else:
        predicted_label = None
        
    
    label_map = {label : i for i, label in enumerate(label_list,1)}
    features = []
    
    ori_sent = []
    ori_label = []
    
    for (ex_index,example) in enumerate(examples):
        
        #import pdb
        #pdb.set_trace()
        
        textlist = example.text_a.split(' ')
        labellist = example.label
        
        ori_sent.append(textlist)
        ori_label.append(labellist)
        
        
        ## annota "O" for these tokens that not be choosen
        '''
        if predicted_label is not None and ex_index > ori_len - 1:
            try:
                assert len(predicted_label[ex_index-iterative_len]) == len(labellist)
            except:
                import pdb
                pdb.set_trace()
                
            for i in range(0, start_end[ex_index-ori_len][0]):
                labellist[i] = 'O' # predicted_label[ex_index][i]
            
            for i in range(start_end[ex_index-ori_len][1], len(labellist)):
                labellist[i] = 'O' # predicted_label[ex_index][i]
        '''
        
        
        
        ##use soft label and negative sampling
        
        if predicted_label is not None and ex_index > iterative_len - 1:
            #import pdb
            #pdb.set_trace()
            start_idx = start_end[ex_index-ori_len][0] - 1
            end_idx = start_end[ex_index-ori_len][1] - 1
            labellist[0:start_idx] = predicted_label[ex_index-iterative_len][0:start_idx]
            labellist[end_idx:-1] = predicted_label[ex_index-iterative_len][end_idx:-1]
            
            idx_list = list(range(1, start_idx+1)) + list(range(end_idx+1, len(labellist)+1))
            #idx_list = idx_list + 1
            sample_num = int(len(idx_list)*0.3)
            sample_idx_list = random.sample(idx_list, sample_num)
        
        
        
        
         
        #if start_end != None:
        #    import pdb
        #    pdb.set_trace()
        
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token) # maybe ##__ will be added
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_ids.append(label_map["[CLS]"])
        
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        
        ntokens.append("[SEP]")
        segment_ids.append(0) # segment ids always
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        
        ##totally mask the labels that do not queried
        if ori_len is not None and ex_index > ori_len - 1:
            
            label_mask = [0] * len(label_ids)
            label_mask[0] = 1
            label_mask[-1] = 1
            
            try:
                #import pdb
                #pdb.set_trace()
                for i in range(start_end[ex_index-ori_len][0], start_end[ex_index-ori_len][1]):
                    label_mask[i] = 1
                
                ##if use soft label and negtive sampling
                if predicted_label is not None and ex_index > iterative_len - 1:
                        for i in sample_idx_list:
                            label_mask[i] = 1
                
                
            except:
                import pdb
                pdb.set_trace()
            
            try:
                assert len(label_mask) == len(label_ids)
            except:
                import pdb
                pdb.set_trace()
            
            #label_mask = [1] * len(label_ids)
            
        else:
            label_mask = [1] * len(label_ids)
        # padding
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        #if ex_index < 1:
        #    logger.info("*** Example ***")
        #    logger.info("guid: %s" % (example.guid))
        #    logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        #    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #    logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #    logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        features.append(InputFeatures(input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_ids,
                        valid_ids=valid,
                        label_mask=label_mask))
    return features, ori_sent, ori_label