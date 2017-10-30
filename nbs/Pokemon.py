
# coding: utf-8

# <img src="https://s3.amazonaws.com/pokemontcg/xy7/54.png" alt="Bulbasaur Pic" style="width: 256px;"/>
# 
# # Pokemon TCG card generator
# 
# - Downloads and saves card data from pokemontcg.io with python sdk, reformats it as YAML
#     - Actually, just use api directly because python sdk is out of date and parts are not compatible with each other
# - Uses keras lstm example to generate card data
# 
# ## Load card data

# In[1]:

# imports
import yaml, json, os, random, requests
from pprint import pprint

data_dir = '/home/ubuntu/fastai-data/pokemon'


# In[3]:

# query pokemontcg api for every card
cards_full = []
for i in range(10):
    response = requests.get('https://api.pokemontcg.io/v1/cards?page={}&pageSize=1000'.format(i+1))
    current_cards = json.loads(response.content)['cards']
    cards_full.extend(current_cards)
    if len(current_cards) < 1000:
        print('-- Cards Loaded ---')
        break
pprint(cards_full[-1])


# In[4]:

# get card data from pokemontcg.io
keys = ['name', 'subtype', 'supertype', 'ability', 'ancient_trait', 'hp', 'evolvesFrom',
        'retreat_cost', 'types', 'attacks', 'weaknesses', 'resistances', 'text']
cards = [{key: card[key] if key in card else None for key in keys} for card in cards_full]


# In[6]:

# save data
with open(os.path.join(data_dir,'cards.json'), 'w+') as f:
     json.dump(cards, f)


# ## Preprocessing
# 
#  - Convert json data to a text representation easy for a character-embedding based model to parse

# In[7]:

# load data
with open(os.path.join(data_dir,'cards.json')) as f:
     cards = json.load(f)
#pprint(cards[-1])


# In[8]:

# augment data
cards = random.sample(cards, len(cards))
#for i in range(2):
#    cards.extend(random.sample(cards, len(cards)))


# In[9]:

# encode card categories as greek letters
alphabet = 'θωερτψυιοπασφγηςκλζχξωβνμ'
# encode type as a unicode character, following https://redd.it/4xvh2q
type_char = '✴☽☽⛩❤✊♨☘☘⚡⛓⚛☔'

types = json.loads(requests.get('https://api.pokemontcg.io/v1/types').content)['types']
types.insert(2, 'Dark')
types.insert(7, 'Green')
subtypes = json.loads(requests.get('https://api.pokemontcg.io/v1/subtypes').content)['subtypes']


# In[10]:

# encode type as unicode character
def type_to_char(t_list):
    if t_list and t_list[0] != 'Free':
        return ''.join([type_char[types.index(t)] for t in t_list])
    else:
        return ''

# convert list of lines to single text, and replaces name with @
def singlify(text, name=None):
    if text:
        text = ''.join(text) if isinstance(text, list) else text
        if name:
            text = text.replace(name, '@')
        return text
    else:
        return ''

# write data as txt file
with open(os.path.join(data_dir,'cards.txt'), 'w+') as f:
    for card in cards:
        lines = ['\n']
        lines.append('|'.join([card['supertype'][0],
                alphabet[subtypes.index(card['subtype'])] if card['subtype'] else '',
                type_to_char(card['types']),
                type_char[types.index(card['weaknesses'][0]['type'])] \
                    + ('^'*int(card['weaknesses'][0]['value'][1]) if '0' in card['weaknesses'][0]['value'] else 'x')\
                    if card['weaknesses'] else '',     
                type_char[types.index(card['resistances'][0]['type'])] \
                    + ('^'*int(card['resistances'][0]['value'][1]) if '0' in card['resistances'][0]['value'] else 'x')\
                    if card['resistances'] else '',     
                '^'*(int(card['hp'])//10) if card['hp'] and card['hp'].isdigit() else '',
                type_to_char(card['retreat_cost']),
                singlify(card['name']), singlify(card['evolvesFrom']), singlify(card['text'],name=card['name'])]))
        if card['ability']:
            lines.append(
                '|'.join(['x', card['ability']['name'],
                          singlify(card['ability']['text'], name=card['name'])]))
        if card['ancient_trait']:
            lines.append(
                '|'.join(['y', card['ancient_trait']['name'],
                          singlify(card['ancient_trait']['text'], name=card['name'])]))
        if card['attacks'] and card['attacks']:
            for attack in card['attacks']:
                lines.append(
                    '|'.join(['z', type_to_char(attack['cost']) if 'cost' in attack else '',
                              str(attack['damage']), singlify(attack['name']),
                              singlify(attack['text'], name=card['name'])]))
        if 'マ' not in ''.join(lines): # no japanese cards
            for line in lines:
                f.write(line+'\n')
            


# ## Create Model

#  - Turn text into embedded sequences that keras can use
#  - Setup model architecture

# In[11]:

# imports
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
import numpy as np


# In[12]:

# load text
path = os.path.join(data_dir,'cards.txt')
text = open(path).read()[:]

print('corpus length:', len(text))
print(text[:128])


# In[13]:

# get characters used in text
chars = sorted(list(set(text)))
vocab_size = len(chars)

print('total chars:', vocab_size)
print(''.join(chars))


# In[14]:

# create character indices
char_indices = dict((c, i) for i, c in enumerate(chars))
# turn text into char indices
idx = [char_indices[c] for c in text]


# In[15]:

maxlen = 128
sentences = []
next_chars = []
for i in range(len(idx)-maxlen+1):
    sentences.append(idx[i: i + maxlen])
    next_chars.append(idx[i+1: i+maxlen+1])


# In[ ]:

print('# of sequences:', len(sentences))

sentences = np.concatenate([[np.array(o)] for o in sentences[:-2]])
next_chars = np.concatenate([[np.array(o)] for o in next_chars[:-2]])


# In[ ]:

np.save(os.path.join(data_dir,'sentences.npy'), sentences)
np.save(os.path.join(data_dir,'next_chars.npy'), sentences)


# In[ ]:

# size of embedding
n_fac = 42


# In[ ]:

# model architecture
model=Sequential([
        Embedding(vocab_size, n_fac, input_length=maxlen),
        GRU(256, input_shape=(n_fac,),return_sequences=True, dropout=0.01, recurrent_dropout=0.01),
        Dropout(0.2),
        GRU(512, return_sequences=True, dropout=0.01, recurrent_dropout=0.01),
        Dropout(0.2),
        TimeDistributed(Dense(vocab_size)),
        Activation('softmax')
    ])
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])
model.summary()


# ## Train Model

# In[ ]:

from numpy.random import choice
import random

# print example text, 
def print_example(length=800, temperature=0.7, mult=2):
    seed_len=40
    path = os.path.join(data_dir,'cards.txt')
    text = open(path).read()[:]
    ind = random.randint(0,len(text)-seed_len-1)
    seed_string = text[ind:ind+seed_len]
    
    for i in range(length):
        if (seed_string.split('\n')[-1].count('|') == 7 or
        seed_string.startswith(('x','y')) and seed_string.split('\n')[-1].count('|') == 1 or
        seed_string.startswith('z') and seed_string.split('\n')[-1].count('|') == 3):
            temp = temperature * mult # make names more creative
        else:
            temp = temperature
        
        x=np.array([char_indices[c] for c in seed_string[-40:]])[np.newaxis,:]
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

result_dir = os.path.join(data_dir, 'results')
weight_path = "weights-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(result_dir, weight_path),
                             monitor='acc', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=2, min_lr=0.00000001)
printer = LambdaCallback(on_epoch_end=print_callback)

callbacks_list = [printer, checkpoint, reduce_lr]


# In[ ]:

num_epochs = 30
history = model.fit(sentences,
                    np.expand_dims(next_chars,-1),
                    batch_size=256,
                    epochs=num_epochs,
                    callbacks=callbacks_list)


# In[ ]:

get_ipython().run_cell_magic(u'capture', u'generated_cards', u'print_example(length=300000, temperature=0.7, mult=2)')


# In[ ]:

with open(os.path.join(data_dir,'cards_generated.txt'), 'w+') as f:
    f.write(generated_cards.stdout)


# ## Process Output
# 
#  - At this point I redid the prior stuff with a premade tensorflow char-rnn model, to see if it was any better. I didn't feel as if there were significant improvements but it did run considerably faster.
#  - Decode generated text back into JSON format
#  - Convert JSON format to card images

# Run in terminal, in tensorflow-char-rnn directory, with python 3.6:
# 
# python train.py --data_file=/home/ubuntu/fastai-data/pokemon/cards.txt --output_dir=/home/ubuntu/fastai-data/pokemon/results_tf --embedding_size=30 --model=lstm --hidden_size=256 --num_layers=3 --batch_size=96 --learning_rate=0.001 --num_epochs=120
# 
# and in a seperate tmux pane:
# 
# tensorboard --logdir=/home/ubuntu/fastai-data/pokemon/results_tf/tensorboard_log --port=6006
# 
# Afterwards, run:
# 
# python sample.py --init_dir=/home/ubuntu/fastai-data/pokemon/results_tf --start_text="P|j|R|g|fx||^^^^^^^^^^|cc|Breloom|" --length=500000 --seed=4745 --temperature=0.7 | tee /home/ubuntu/fastai-data/pokemon/cards_generated_tf2.txt

# In[ ]:

# encode card categories as greek letters
alphabet = 'θωερτψυιοπασφγηςκλζχξωβνμ'
# encode type as a unicode character, following https://redd.it/4xvh2q
type_char = '✴☽⛩❤✊♨☘⚡⛓⚛☔'

types = json.loads(requests.get('https://api.pokemontcg.io/v1/types').content)['types']
subtypes = json.loads(requests.get('https://api.pokemontcg.io/v1/subtypes').content)['subtypes']
supertypes = json.loads(requests.get('https://api.pokemontcg.io/v1/supertypes').content)['supertypes']
with open(os.path.join(data_dir,'cards.json')) as f:
     old_names = [card['name'] for card in json.load(f)]


# In[ ]:

# decode type from unicode character
def char_to_type(chars):
    if chars and len(chars) > 0:
        return [types[type_char.index(char)] for char in chars]
    else:
        return None

cards = []
card = None
with open(os.path.join(data_dir,'cards_generated.txt')) as f:
    for line in f:
        line = line.split('|')
        if line[0] in ('P','E','T'):
            if card and card['name'].rstrip() not in old_names:
                cards.append(card)
            try:
                card = {'supertype': supertypes[('P','E','T').index(line[0])],
                        'subtype': subtypes[alphabet.index(line[1])] if line[1] else None,
                        'types': char_to_type(line[2]),
                        'weaknesses':
                        {'type': types[type_char.index(line[3][0])],
                         'value': '×2' if line[4][1] == 'x' else '-'+str(len(line[4])-1)+'0'} if line[4] else None,
                        'resistances':
                        {'type': types[type_char.index(line[4][0])],
                         'value': '×2' if line[5][1] == 'x' else '-'+str(len(line[5])-1)+'0'} if line[5] else None,
                        'hp': len(line[5])*10 if line[6] else None,
                        'retreat_cost': char_to_type(line[6]),
                        'name': line[7].rstrip(),
                        'evolvesFrom': line[8].rstrip(),
                        'text': line[9].replace('@',line[8]).rstrip() if len(line) > 9 else None}
            except:
                card = None
                print('Skipped card')
        elif line[0] == 'x' and card and card['supertype'] == 'Pokémon':
            try:
                card['ability'] = {'name':line[1].rstrip(),
                                   'text':line[2].replace('@',card['name']).rstrip() if len(line) > 2 else None}
            except:
                print('Skipped ability')
        elif line[0] == 'y' and card and card['supertype'] == 'Pokémon':
            try:
                card['ancient_trait'] = {'name':line[1].rstrip(),
                                         'text':line[2].replace('@',card['name']).rstrip() if len(line) > 2 else None}
            except:
                print('Skipped trait')
        elif line[0] == 'z' and card and card['supertype'] == 'Pokémon':
            try:
                card.setdefault('attacks', []).append(
                    {'cost': char_to_type(line[1]),
                     'damage': line[2],
                     'name': line[3].rstrip(),
                     'text': line[4].replace('@',card['name']).rstrip() if len(line) > 4 else None})
            except:
                print('Skipped attack')
                                     


# In[ ]:

class ExplicitDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True
    
with open('cards_generated.yml', 'w+') as f:
     yaml.dump(cards, f, allow_unicode=True, Dumper=ExplicitDumper, default_flow_style=False)


# In[ ]:

from IPython.display import FileLink
FileLink('cards_generated.yml')


# ## Create Card Mockups
# 
# 1. Put [Basic](https://paulsnoops.deviantart.com/art/BWXY-Basic-Pokemon-blanks-734x1024-601940663) and [Trainer & Energy](https://paulsnoops.deviantart.com/art/BWXY-Trainer-and-Energy-blanks-734x1024-601953321) templates in template_path, in appropriate hierarchy.
#     - For pokemon, first have folder with supertype name, then folders with subtype name underneath, then images with name {type}.png
#     - For trainers and energy, first have folder with supertype name, then images with name {subtype}.png
#     - If subtype has space replace with underscore
# 2. Put [symbols.png](https://paulsnoops.deviantart.com/art/BWXY-Symbol-Sheet-601935489) and [holosheet.png](https://aschefield101.deviantart.com/art/XY-HoloSheet-Japanese-417932199) under template_path
# 2. Put [font collection](http://www.pokebeach.com/forums/threads/faking-resources-and-help-designing-original-tcg-cards.128741/) in fonts folder under template_path
#     - Rename fonts in abbreviated form, as shown in code below

# In[ ]:

from PIL import Image, ImageDraw, ImageFont
import os, textwrap

data_dir = '/home/ubuntu/fastai-data/pokemon'
template_path = os.path.join(data_dir, 'templates')
font_path = os.path.join(template_dir, 'fonts')
save_path = os.path.join(data_dir, 'card_results')


# In[ ]:

import yaml, pprint, re, unidecode

card_data = []
with open('cards_generated.yml') as f:
     card_data = yaml.load(f)


# In[ ]:

# only supports basic pokemon for now
def get_energy_img(energy, category):
    energies = ['Grass', 'Fire', 'Water', 'Electric', 'Psychic', 'Fighting',
                'Dark', 'Metal', 'Fairy', 'Dragon', 'Colorless']
    full_img = Image.open(os.path.join(template_path,'symbols.png'))
    if category is 'attack':
        img = full_img.crop((46+energies.index(energy)*57, 85, 85+energies.index(energy)*57, 135))
    if category is 'weakness':
        img = full_img.crop((50+energies.index(energy)*57, 210, 80+energies.index(energy)*57, 250))
    return img

def gen_card_img(card):
    if card['supertype'] == 'Pokémon':
        img = Image.open(os.path.join(template_path,
                         card['supertype'], card['subtype'].replace(' ', '_'),
                         card['types'][0]+'.png'))
        
        d = ImageDraw.Draw(img)
        
        f = ImageFont.truetype(font=os.path.join(font_path,'gill-rb.ttf'), size=48)
        d.text((180,36), card['name'], font=f, fill='black')

        f = ImageFont.truetype(font=os.path.join(font_path,'gill-rb.ttf'), size=18)
        d.text((556, 68), 'HP', font=f, fill='black')

        f = ImageFont.truetype(font=os.path.join(font_path,'futura-cb.ttf'), size=44)
        d.text((582, 42), str(card['hp']), font=f, fill='black')
        
        f = ImageFont.truetype(font=os.path.join(font_path,'futura-cb.ttf'), size=30)
        if card['weaknesses']:
            energy_img = get_energy_img(card['weaknesses']['type'], 'weakness')
            img.paste(energy_img, (65, 888), energy_img)
            d.text((100, 890), card['weaknesses']['value'], font=f, fill='black')
        if card['resistances']:
            energy_img = get_energy_img(card['resistances']['type'], 'weakness')
            img.paste(energy_img, (195, 888), energy_img)
            d.text((230, 890), card['resistances']['value'], font=f, fill='black')
        
        full_img = Image.open('symbols.png')
        retreat_img = full_img.crop((517, 433, 517+32*len(card['retreat_cost']),463))
        img.paste(retreat_img, (150, 938), retreat_img)

        start_height = 560
        if 'ability' in card:
            ability = card['ability']
            
            ability_img = full_img.crop((50, 433, 212, 475))
            img.paste(ability_img, (60, start_height+5), ability_img)
            
            f = ImageFont.truetype(font=os.path.join(font_path,'gill-cb.ttf'), size=44)
            d.text((240, start_height), ability['name'], font=f, fill='#c23600')
            
            f = ImageFont.truetype(font=os.path.join(font_path,'gill-rp.ttf'), size=30)
            d.multiline_text((60, start_height+54), textwrap.fill(ability['text'], width=48), font=f, fill='black')
            
            start_height += 80 + d.multiline_textsize(textwrap.fill(ability['text'], width=48), font=f)[1]
        if 'attacks' in card:
            for attack in card['attacks']:
                if start_height >= 760:
                    break
                
                for n in range(len(attack['cost'])):
                    energy_img = get_energy_img(attack['cost'][n],'attack')
                    img.paste(energy_img, (60+n*45, start_height), energy_img)
                
                f = ImageFont.truetype(font=os.path.join(font_path,'gill-cb.ttf'), size=44)
                d.text((115+n*45, start_height), attack['name'], font=f, fill='black')

                f = ImageFont.truetype(font=os.path.join(font_path,'futura-cb.ttf'), size=44)
                d.text((612, start_height), attack['damage'], font=f, fill='black')

                f = ImageFont.truetype(font=os.path.join(font_path,'gill-rp.ttf'), size=30)
                d.multiline_text((60, start_height+54), textwrap.fill(attack['text'], width=48), font=f, fill='black')

                start_height += 80 + d.multiline_textsize(textwrap.fill(attack['text'], width=48), font=f)[1]
        
    elif card['supertype'] == 'Trainer':
        img = Image.open(os.path.join(template_path,
            card['supertype'], (card['subtype'].replace(' ','_') if card['subtype'] else 'Supporter')+'.png'))
        d = ImageDraw.Draw(img)
        
        f = ImageFont.truetype(font=os.path.join(font_path,'gill-rb.ttf'), size=44)
        d.text((85,105), card['name'], font=f, fill='black')
        
        f = ImageFont.truetype(font=os.path.join(font_path,'gill-rp.ttf'), size=30)
        d.multiline_text((95, 570), textwrap.fill(card['text'] if card['text'] else '', width=42), font=f,fill='black')
    else:
        img = Image.open(os.path.join(template_path,
            card['supertype'], (card['subtype'].replace(' ','_') if card['subtype'] else 'Special')+'.png'))
        d = ImageDraw.Draw(img)
        
        f = ImageFont.truetype(font=os.path.join(font_path,'gill-rb.ttf'), size=30)
        d.text((80,100), 'Special Energy', font=f, fill='black')
        
        f = ImageFont.truetype(font=os.path.join(font_path,'gill-rb.ttf'), size=40)
        d.text((60,655), card['name'], font=f, fill='black')
        
        f = ImageFont.truetype(font=os.path.join(font_path,'gill-rp.ttf'), size=30)
        d.multiline_text((60, 720), textwrap.fill(card['text'], width=48), font=f, fill='black')
    
    img.thumbnail((590,590))
    background = Image.open(os.path.join(template_path,'holosheet.jpg'))
    background.paste(img, (0, 0), img)
    img = background

    return img


# In[ ]:

# turns string into filename
def slugify(value):
    value = unidecode.unidecode(value)
    value = str(re.sub('[^\w\s-]', '', value).strip())
    value = str(re.sub('[-\s]+', '-', value))
    return value


# In[ ]:

for card in card_data:
    try:
        print(card['name'])
        img = gen_card_img(card)
        img.save(os.path.join(save_path, slugify(card['name'])+'.jpg'))
    except:
        print('skipped a card')

