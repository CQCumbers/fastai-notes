
# coding: utf-8

# ![Header](https://img00.deviantart.net/32dd/i/2015/117/7/c/redvsblue_chubbs_by_feathernotes-d461960.jpg)
# 
# # Red vs. Blue Dialogue Generator (Sarge Chatbot)
# 
# 1. Scrape RvB transcripts from [RoosterTooths](http://roostertooths.com/transcripts.php)
# 2. Train word based LSTM on scripts, starting from pretrained embeddings
# 3. Predict Sarge dialogue by priming with the conversor's dialogue added to some random dialogue
# 
# ## Scrape Transcripts
# 
#  - Create empty scripts.txt file in appropriate directory beforehand

# In[8]:


from lxml import html
import requests, os

path = '/home/ubuntu/fastai-data/rvb/scripts.txt'
path1 = '/home/ubuntu/fastai-data/rvb/scripts1.txt'
path2 = '/home/ubuntu/fastai-data/rvb/scripts2.txt'


# In[2]:


# scrape rooster tooths for RvB scripts
def scrape():
    with open(path1, 'w') as f:
        for i in [x for x in range(347)]:
            page = requests.get('http://roostertooths.com/transcripts.php?eid={}'.format(i+1))
            tree = html.fromstring(page.content)
            lines = []
            f.write('\n\n'+tree.xpath('//p[@class="breadcrumbs"]/a//text()')[1]
                  +'\n'+tree.xpath('//h1//text()')[0]+'\n\n')
            for row in tree.xpath('//table[@class="script"]/tr'):
                f.write(''.join(row.xpath('.//td//text()'))+'\n')

scrape()


# In[66]:


# scrate seinology for seinfeld scripts, to augment data
def scrape2():
    with open(path2, 'w') as f:
        for i in [x for x in range(100)]:
            page = requests.get('http://www.seinology.com/scripts/script-{}.shtml'.format(70+i))
            if page.status_code == 200:
                tree = html.fromstring(page.content)
                lines = []
                f.write('\n\n'+tree.xpath('//p//text()')[27]+'\n')
                f.write('\n'.join([l.strip().replace(': ', ':') for l in ''.join(tree.xpath('//p//text()')).split(
                    '============')[-1].split('\n')
                    if (len(l.strip()) > 0 and not l.strip()[0] in ['(','[','='])]))
            else:
                print('script '+str(70+i)+' not found')

scrape2()


# ## Prepare Text

# In[43]:


# imports
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Nadam
import numpy as np
from IPython.display import FileLink


# In[80]:


# load text
text1 = open(path1).read()[:]
text2 = open(path2).read()[:]
text = text1 + '\n' + text2
print('corpus 1 length:', len(text1))
print('corpus 2 length:', len(text2))
print('corpus length:', len(text))
print('')

lines = []
for line in text.split('\n'):
    # only get spoken lines from RvB
    if line.startswith(' '):
        line = line[1:]
        # remove last line if captioned
        if line.startswith('caption'):
            line = lines[-1].split(':',1)[0].upper() + ':' + line.split(':',1)[1].lower()
            lines = lines[:-1]
        else:
            line = line.split(':',1)[0].upper() + ':' + line.split(':',1)[1].lower()
        lines.append(line)
    # only get spoken lines from seinfeld
    elif len(line.split(':',1)) == 2 and line.split(':',1)[0].isupper():
        line = line.split(':',1)[0].upper() + ':' + line.split(':',1)[1].lower()
        lines.append(line)
        
replacemap = {"\x91": '"', "\x93": '"', "\x92": "'", "\x94": "'",
              '[': '(', ']': ')', '\x85': '\n', '\xa0': ' ', '\x96': ''}
text = '\n'.join(lines)
for k, v in replacemap.items():
    text = text.replace(k, v)

with open(path, 'w') as f:
    f.write(text)


# In[81]:


chars = sorted(list(set(text)))
print(chars)
vocab_size = len(chars)
print('total chars:', vocab_size)

# create character embeddings
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

idx = [char_indices[c] for c in text]


# In[82]:


maxlen = 64
sentences = []
next_chars = []
for i in range(len(idx)-maxlen+1):
    sentences.append(idx[i: i + maxlen])
    next_chars.append(idx[i+1: i+maxlen+1])


# In[83]:


print('nb sequences:', len(sentences))
print('nb chars:', len(next_chars))


# In[ ]:


sentences = np.concatenate([[np.array(o)] for o in sentences[:-2]])
next_chars = np.concatenate([[np.array(o)] for o in next_chars[:-2]])


# In[ ]:


n_fac = 128


# ## Train model

# In[ ]:


from keras.models import Sequential
from keras.layers import *

# 2 layer GRU network with 256 and 512 channels
model = Sequential([
    Embedding(vocab_size, n_fac, input_length=maxlen),
    CuDNNGRU(512, input_shape=(n_fac,), return_sequences=True),
    Dropout(0.5),
    CuDNNGRU(256, return_sequences=True),
    Dropout(0.5),
    TimeDistributed(Dense(vocab_size)),
    Activation('softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=Nadam(lr=1e-4), metrics=['acc'])
model.summary()


# In[ ]:


# save as JSON
json_string = model.to_json()
with open('results/rvb-model.json', 'w+') as f:
    f.write(json_string)
FileLink('results/rvb-model.json')


# In[ ]:


from numpy.random import choice
import random

def print_example(length=800, temp=0.8):
    seed_len=maxlen
    text = open(path).read().lower()[:]
    ind = random.randint(0,len(text)-seed_len-1)
    seed_string = text[ind:ind+seed_len]
    for i in range(length):
        x=np.array([char_indices[c] for c in seed_string[-seed_len:]])[np.newaxis,:]
        preds = model.predict(x, verbose=0)[0][-1]
        preds = np.log(preds) / temp
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        next_char = choice(chars, p=preds)
        print(next_char, end="")
        seed_string = seed_string + next_char
    #print(seed_string[seed_len:])


# In[ ]:


from keras.callbacks import *

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


# In[ ]:


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
import h5py

def print_callback(logs, epoch):
    print_example()

weight_dir = '/home/ubuntu/fastai-data/rvb/weights'
weight_path = "weights-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(weight_dir, weight_path),
                             monitor='acc', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=1, min_lr=0.000001)
printer = LambdaCallback(on_epoch_end=print_callback)
clr = CyclicLR(base_lr=0.0001, max_lr=0.001, step_size=2000., mode='triangular')

callbacks_list = [printer, checkpoint, reduce_lr, clr]


# In[ ]:


num_epochs = 10
model.load_weights(os.path.join(weight_dir, 'weights-01.hdf5'))
history = []
history.append(model.fit(sentences,
                    np.expand_dims(next_chars,-1),
                    shuffle=True,
                    batch_size=64,
                    epochs=num_epochs,
                    validation_split=0.15,
                    callbacks=callbacks_list))


# - I like how it learns Spanish exclusively from Lopez's dialogue

# In[ ]:


print_example(length=20000)

