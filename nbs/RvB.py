
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

# In[1]:


from lxml import html
import requests, os

path = '/home/ubuntu/fastai-data/rvb/scripts.txt'


# In[2]:


with open(path, 'w') as f:
   for i in range(347):
       page = requests.get('http://roostertooths.com/transcripts.php?eid={}'.format(i+1))
       tree = html.fromstring(page.content)
       lines = []
       f.write('\n\n'+tree.xpath('//p[@class="breadcrumbs"]/a//text()')[1]
             +'\n'+tree.xpath('//h1//text()')[0]+'\n\n')
       for row in tree.xpath('//table[@class="script"]/tr'):
           f.write(''.join(row.xpath('.//td//text()'))+'\n')


# ## Prepare Text

# In[3]:


# imports
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
import numpy as np
from IPython.display import FileLink


# In[4]:


# load text
text = open(path).read().lower()[:]
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print(chars)
vocab_size = len(chars)
print('total chars:', vocab_size)

# create character embeddings
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

idx = [char_indices[c] for c in text]


# In[5]:


maxlen = 128
sentences = []
next_chars = []
for i in range(len(idx)-maxlen+1):
    sentences.append(idx[i: i + maxlen])
    next_chars.append(idx[i+1: i+maxlen+1])


# In[ ]:


print('nb sequences:', len(sentences))
print('nb chars:', len(next_chars))


# In[ ]:


sentences = np.concatenate([[np.array(o)] for o in sentences[:-2]])
next_chars = np.concatenate([[np.array(o)] for o in next_chars[:-2]])


# In[ ]:


n_fac = 100


# ## Train model

# In[ ]:


from keras.models import Sequential
from keras.layers import *

# 2 layer GRU network with 256 and 512 channels
model = Sequential([
    Embedding(vocab_size, n_fac, input_length=maxlen),
    CuDNNLSTM(512, input_shape=(n_fac,), return_sequences=True),
    Dropout(0.01),
    CuDNNLSTM(512, return_sequences=True),
    Dropout(0.01),
    TimeDistributed(Dense(vocab_size)),
    Activation('softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])
model.summary()


# In[ ]:


# save as JSON
json_string = model.to_json()
with open('grifbot_model.json', 'w+') as f:
    f.write(json_string)
FileLink('grifbot_model.json')


# In[ ]:


from numpy.random import choice
import random

def print_example(length=800, temp=0.75):
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


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
import h5py

def print_callback(logs, epoch):
    print_example()

weight_dir = '/home/ubuntu/fastai-data/rvb/weights'
weight_path = "weights-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(weight_dir, weight_path),
                             monitor='acc', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=3, min_lr=0.000001)
printer = LambdaCallback(on_epoch_end=print_callback)

callbacks_list = [printer, checkpoint, reduce_lr]


# In[ ]:


num_epochs = 7
model.load_weights(os.path.join(weight_dir, 'weights-05.hdf5'))
history = []
history.append(model.fit(sentences,
                    np.expand_dims(next_chars,-1),
                    batch_size=128,
                    epochs=num_epochs,
                    callbacks=callbacks_list))


# - I like how it learns Spanish exclusively from Lopez's dialogue

# In[ ]:


print_example(length=20000)

