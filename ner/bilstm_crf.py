import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score
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

epochs = 10
MAXLEN = 200
crf_lr_multiplier = 1000
batch_size = 32
learning_rate = 3e-4

dict_path = '../pretrain_model/chinese_roberta/vocab.txt'

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

train_data = load_data('./data/example.train')[:100]
valid_data = load_data('./data/example.dev')
test_data = load_data('./data/example.test')

tokenizer = Tokenizer(token_dict=dict_path,do_lower_case=True)

# text = '我爱中国'
# temp = tokenizer.tokenize(text)
# print(temp)

# exit(0)
#类别映射
labels = ['PER','LOC','ORG']
id2label = dict(enumerate(labels))
label2id = {j:i for i,j in id2label.items()}
num_labels = len(labels)*2 +1 #tag*BI + O

class data_generator(DataGenerator):

    def __iter__(self,random=False):
        batch_token_ids, batch_labels = [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < MAXLEN:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l]*2 + 1
                        I = label2id[l]*2 + 2
                        labels += ([B] + [I]*(len(w_token_ids)-1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            batch_token_ids.append(token_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids,length=MAXLEN)
                batch_labels = sequence_padding(batch_labels,length=MAXLEN)
                yield batch_token_ids, batch_labels
                batch_token_ids, batch_labels = [], []

model = keras.models.Sequential()
model.add(keras.layers.Embedding(21129,1024,input_length=MAXLEN))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(1024,return_sequences=True)))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(1024,return_sequences=True)))
model.add(Dense(num_labels))
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = model.output
output = CRF(output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learning_rate),
    metrics=[CRF.sparse_accuracy]
)


class nameentityrecognizer(ViterbiDecoder):

    def recognize(self, tokens_ids, y_true):
        nodes = model.predict(tokens_ids)[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0 :
                if label %2 ==1:
                    starting = True
                    entities.append([[i], id2label[(label-1)//2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], 1)
                for w, l in entities]

NER = nameentityrecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

def evaluate(data):

    X, Y, Z = 1e-10, 1e-10, 1e-10
    for x_true,y_true in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(x_true,y_true))
        T = set([tuple(i) for i in d if i[1] !='O'])
        X +=len(R & T)
        Y +=len(R)
        Z +=len(T)
    f1, precision, recall = 2*X/(Y+Z), X/Y, X/Z

    return f1, precision, recall

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        print(NER.trans)
        f1, precision, recall = evaluate(data_generator(valid_data,batch_size))
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_model.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


TRAIN = 1

if __name__ == '__main__':


    if TRAIN:
        evaluator = Evaluator()
        train_generator = data_generator(train_data, batch_size)

        # for b in train_generator:
        #     t_ids, labels = b[0],b[1]
        #     print(t_ids.shape)
        #     print(t_ids)
        #     break

        model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator]
        )


    else:
        model.load_weights('./best_model.weights')
        NER.trans = K.eval(CRF.trans)