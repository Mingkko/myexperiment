import numpy as np
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import jieba

def load_data(filename):
    D = []

    with open(filename,encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ""
            for c in l.split('\n'):
                char, this_flag = c.split(' ')
                if this_flag=='O' and last_flag == 'O':
                    d[-1][0] +=char
                elif this_flag=='O' and last_flag != 'O':
                    d.append([char,'O'])
                elif this_flag[0] == 'B':
                    d.append([char,this_flag[2:]])
                else:
                    d[-1][0] +=char
                last_flag = this_flag
            D.append(d)

    return D

train_data = load_data('./data/example.train')
dev_data = load_data('./data/example.dev')
test_data = load_data('./data/example.test')

tokenizer = Tokenizer(token_dict=)

text = '我爱中国'
temp = tokenizer.tokenize(text)
print(temp)

exit(0)
#类别映射
labels = ['PER','LOC','ORG']
id2label = dict(enumerate(labels))
label2id = {j:i for i,j in id2label.items()}
num_labels = len(labels)*2 +1 #tag*BI + O

class data_generater(DataGenerator):

    def __init__(self,random=False):
        batch_token_ids, batch_labels = [], []
        for is_end, item in self.sample(random):
            break
