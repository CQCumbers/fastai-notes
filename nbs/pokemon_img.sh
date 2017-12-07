#!/bin/sh

# install necessary programs
sudo apt-get install -y unzip imagemagick
sudo -H pip3 install imagehash requests pillow
echo 'installed programs'

# download zipped gifs from pkparaiso
mkdir -p ~/fastai-data/pokemon_img
cd ~/fastai-data/pokemon_img
wget https://www.pkparaiso.com/xy/xy-1291-animated-gifs-7bdab16fdede0fa327e2c04a6545ea4f.zip -O gifs.zip
# unzip download
unzip -q gifs.zip -d gifs
echo 'extracted gifs'

# split gif into pngs
mkdir pngs
find gifs -name '*.gif' | xargs -n1 -P8 sh -c 'convert $0 pngs/$(basename $0 .gif)-%02d.png'
echo 'split into pngs'

# whiten background and trim
find pngs -name '*.png' | xargs -n1 -P8 sh -c 'convert $0 -background white -alpha remove -trim +repage -thumbnail 122x122 -gravity center -extent 128x128 $0'
echo 'trimmed and whitened'

# run python script
cat << EOF > pokescript.py
import glob, shutil, json, collections, os
import requests, imagehash
from PIL import Image

data_dir = '/home/ubuntu/fastai-data/pokemon_img'

for gif in glob.glob(os.path.join(data_dir, 'gifs', '*.gif')):
    name = gif.split('/')[-1][:-4].split('-')[0]
    response = requests.get('https://api.pokemontcg.io/v1/cards?name={}&supertype=pokemon'.format(name))
    hashes = []
    try:
        types = json.loads(response.content.decode('utf-8'))['cards'][-1]['types'][0]
        os.makedirs(os.path.join(data_dir, 'train', types), exist_ok=True)
        for png in glob.glob(os.path.join(data_dir, 'pngs', '{}*.png'.format(name))):
            _hash = imagehash.dhash(Image.open(png))
            if _hash not in hashes:
                hashes.append(_hash)
                shutil.move(png, os.path.join(data_dir, 'train', types))
    except (KeyError, IndexError):
        print('Error on '+gif)

EOF
sudo chmod 755 pokescript.py
sudo python3 ./pokescript.py
rm -f pokescript.py
echo 'sorted with python script'

# create other folders
mkdir results
mkdir temp
mkdir train
echo 'finished creating folders'
