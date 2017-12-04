
# coding: utf-8

# In[30]:


import glob, shutil, json, collections
import requests, imagehash
from PIL import Image

data_dir = '/home/ubuntu/fastai-data/pokemon_img'

# In[1]:


# imports
from itertools import product

from keras import backend as K
from keras.models import Model
from keras.initializers import *
from keras.layers import *
from keras.engine.topology import Layer
from keras.optimizers import Adam
import itertools, os
import numpy as np


# In[70]:


# length of noise vector
z_length = 100
# number of pokemon types
num_types = 11

# generator network architecture
def get_generator(x, y, dim=16, depth=64):
    y = Dense(z_length)(y)
    x = Multiply()([x, y]) # combine noise and type
    x = Dense(dim*dim*depth)(x)
    x = Reshape((dim, dim, depth))(x)
   
    x = Conv2DTranspose(64, 4, strides=4, padding='same')(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2DTranspose(32, 4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2DTranspose(3, 4, strides=2, padding='same')(x)
    x = Activation('tanh')(x)
    
    return x

# discriminator network architecture
def get_discriminator(x):
    x = Conv2D(32, 4, padding='same', strides=2)(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(64, 3, padding='same', strides=2)(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(128, 3, padding='same', strides=2)(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(128, 3, padding='same', strides=2)(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    y = Dense(11)(x) # classify type of pokemon
    y = Activation('softmax')(y)
    x = Dense(1)(x) # classify real or fake
    x = Activation('sigmoid')(x)
    
    return (x, y)


# In[71]:


g_input = Input(shape=[z_length], name='noise')
g_input2 = Input(shape=[num_types], name='type')
generator = Model(inputs=[g_input, g_input2], outputs=get_generator(g_input, g_input2))

g_test = generator.predict({'noise': np.random.normal(0, 1, size=(8, z_length)),
                            'type': np.eye(num_types)[np.random.choice(num_types, 8)]})
print('max: {} min: {} mean: {}'.format(np.max(g_test), np.min(g_test), np.mean(g_test)))


# In[72]:


d_input = Input(shape=[256, 256, 3], name='images')
discriminator = Model(inputs=d_input, outputs=get_discriminator(d_input))

d_test, d_test2 = discriminator.predict({'images': g_test})
print('max: {} min: {} mean: {}'.format(np.max(d_test), np.min(d_test), np.mean(d_test)))


# In[73]:


batch_size = 64



# discriminator_model trains discriminator with real and generated images
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

# blog article suggests perturbing in all directions (-1 to 1 rather than 0 to 1)
def perturb(input, c=0.1):
    b, row, col, k = batch_size, 256, 256, 3 #input.shape
    alpha = K.repeat_elements(K.repeat_elements(K.repeat_elements(
        K.random_uniform((b, 1, 1, 1), 0, 1), row, 1), col, 2), k, 3)
    x_hat = alpha*input + (1-alpha)*(input + c * K.std(input) * K.random_uniform((b, row, col, k), 0, 1))
    return x_hat

# inputs for stacked generator/discriminator
imgs = Input(shape=[256, 256, 3]) # real mini-batch
p_imgs = Lambda(perturb, output_shape=K.int_shape(imgs)[1:])(imgs) # perturbed mini-batch
noise = Input(shape=[z_length])
types = Input(shape=[num_types])

# from keras WGAN-GP, though called with randomly perturbed inputs rather than averaged inputs
def gradient_penalty(y_true, y_pred, x_hat=p_imgs):
    gradients = K.gradients(K.sum(y_pred), x_hat)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients), axis=[1,2,3]))
    gradient_penalty = K.mean(K.square(gradient_l2_norm - 1))
    return gradient_penalty

# discriminator model with inputs both real and generated images
discriminator_model = Model(inputs=[imgs, noise, types],
                    outputs=discriminator(imgs)+discriminator(generator([noise, types]))+discriminator(p_imgs))
discriminator_model.compile(optimizer=Adam(3e-4),
                    loss=['binary_crossentropy', 'categorical_crossentropy',
                          'binary_crossentropy', 'categorical_crossentropy',
                          gradient_penalty, gradient_penalty],
                    loss_weights=[1.8, 0.2, 0.9, 0.1, 1.0, 0])

#test2 = discriminator_model.predict([generator.predict(np.random.normal(size=(4, 128))),
#                                     np.random.normal(size=(4, 128))])
#print('max: {} min: {} mean: {}'.format(np.max(test2[1]), np.min(test2[1]), np.mean(test2[1])))
discriminator_model.summary()


# In[74]:


# generator_model trains generator to create image, optimizes to maximize discriminator loss
for layer in discriminator.layers:
    layer.trainable = False
for layer in generator.layers:
    layer.trainable = True
discriminator.trainable = False
generator.trainable = True

generator_model = Model(inputs=[noise, types], outputs=discriminator(generator([noise, types])))
generator_model.compile(optimizer=Adam(1e-3),
                        loss=['binary_crossentropy', 'categorical_crossentropy'],
                        loss_weights=[0.9, 0.1])

#test = generator_model.predict(np.random.normal(size=(4, 256)))
#print('max: {} min: {} mean: {}'.format(np.max(test), np.min(test), np.mean(test)))
generator_model.summary()


# In[37]:


from keras.preprocessing import image
#from matplotlib import pyplot as plt
import numpy as np
import random, os

# Instantiate plotting tool
#%matplotlib inline


# In[38]:


def plots(ims, figsize=(12,6), rows=2, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


# In[39]:


data_dir = '/home/ubuntu/fastai-data/pokemon_img'
train_path = os.path.join(data_dir, 'train')
temp_path = os.path.join(data_dir, 'temp')
result_path = os.path.join(data_dir, 'results')


# In[40]:


def get_batches(dirname, temp_dir=None, shuffle=True, batch_size=batch_size):
    gen = image.ImageDataGenerator(preprocessing_function=lambda x: (x - 226)/57,
                                  horizontal_flip=True,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  zoom_range=[1.0, 1.65],
                                  fill_mode='constant',
                                  cval=255)
    return gen.flow_from_directory(dirname,
                                  target_size=(256,256),
                                  class_mode='categorical',
                                  color_mode='rgb',
                                  shuffle=shuffle,
                                  save_to_dir=temp_dir,
                                  batch_size=batch_size)

#batches = get_batches(train_path, temp_dir=temp_path)
batches = get_batches(train_path)

batch, labels = next(batches)
print('mean: {} std dev: {}'.format(np.mean(batch), np.std(batch)))
#plots([image.load_img(os.path.join(temp_path, img)) for img in random.sample(os.listdir(temp_path), 8)])


# In[41]:


from PIL import Image

def tile_images(image_stack):
    assert len(image_stack.shape) == 4
    image_list = [image_stack[i, :, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images

def generate_images(generator, output_dir, epoch):
    test_image_stack = generator.predict([np.random.normal(0, 1, size=(8, z_length)), np.eye(num_types)[:8]]) 
    test_image_stack = (test_image_stack * 57) + 226
    test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
    tiled_output = tile_images(test_image_stack)
    tiled_output = Image.fromarray(tiled_output, mode='RGB')
    outfile = os.path.join(output_dir, 'epoch_{}.png'.format(epoch))
    tiled_output.save(outfile)


# In[42]:


sample_period = int(38376 / batch_size / 4) # sample images and print losses every 1/4 epoch
save_period = int(38376 / batch_size / 20) # save weights every 1/20 epoch
true_positive_y = np.ones((batch_size, 1))
true_negative_y = np.zeros((batch_size, 1))
dummy_y = np.zeros((batch_size, 1)) + 0.5

epoch = 0
if epoch > 0:
    generator_model.load_weights(os.path.join(result_path, 'gan_g_weights{}.h5'.format(epoch)))
    discriminator_model.load_weights(os.path.join(result_path, 'gan_d_weights{}.h5'.format(epoch)))
    """
    discriminator_loss = np.loadtxt(os.path.join(result_path, 'gan_d_loss_history.csv'))
    discriminator_loss = list(np.broadcast_to(
        np.expand_dims(discriminator_loss, axis=1),(discriminator_loss.shape[0], 4)))
    generator_loss =  list(np.loadtxt(os.path.join(result_path, 'gan_g_loss_history.csv')))
    batch_num = len(discriminator_loss) + 1
    """
    discriminator_loss = []
    generator_loss = []
    batch_num = 0
else:
    discriminator_loss = []
    generator_loss = []
    batch_num = 0

epoch += 1
print('Epoch ' + str(epoch) + '\n')
for imgs, types in batches:
    
    # save weights and generate sample images every so often
    if batch_num % save_period == 0 and batch_num > 1:
        print('D. Loss | Total: ' + str(discriminator_loss[-1][0])
              + ' | Real: ' + str(discriminator_loss[-1][1])
              + ' | Fake: ' + str(discriminator_loss[-1][3])
              + ' | Penalty: ' + str(discriminator_loss[-1][5])
              + ' | Categorical: '+str(discriminator_loss[-1][2]))
        np.savetxt(os.path.join(result_path, 'gan_d_loss_history.csv'), np.asarray(discriminator_loss)[:,0])
        
        print('G. Loss | Total: ' + str(generator_loss[-1][0])
             + ' | Real/Fake: ' +  str(generator_loss[-1][1])
             + ' | Categorical: ' + str(generator_loss[-1][2]))
        np.savetxt(os.path.join(result_path, 'gan_g_loss_history.csv'), np.asarray(generator_loss)[:,0])
        
    if batch_num % sample_period == 0 and batch_num > 1:
        generate_images(generator, result_path, epoch)
        if discriminator_loss[-1][0] < 2.0 and generator_loss[-1][0] < 3.5:
            try:
                generator_model.save_weights(os.path.join(result_path, 'gan_g_weights{}.h5'.format(epoch)))
                discriminator_model.save_weights(os.path.join(result_path, 'gan_d_weights{}.h5'.format(epoch)))
            except:
                print('Weights could not be saved')
        else:
            print('Loss was too high - weights not saved')
            generator_model.optimizer.lr *= 0.3
            discriminator_model.optimizer.lr *= 0.3
            
        epoch += 1
        print('\nEpoch ' + str(epoch))
    
    if len(imgs) == batch_size:
        # smooth positive labels only
        positive_y = np.random.normal(1, 0.2, size=(batch_size, 1)) + 1
        negative_y = np.zeros((batch_size, 1))
    
        # train discriminator with real, generated, and perturbed images
        noise = np.random.normal(0, 1, size=(batch_size, z_length))
        gen_types = np.eye(num_types)[np.random.choice(num_types, batch_size)]
        discriminator_loss.append(
            discriminator_model.train_on_batch([imgs, noise, gen_types],
                                               [positive_y, types, negative_y, gen_types, dummy_y, gen_types]))

        # train generator to maximize discriminator loss
        noise2 = np.random.normal(0, 1, size=(batch_size, z_length))
        gen_types2 = np.eye(num_types)[np.random.choice(num_types, batch_size)]
        generator_loss.append(
            generator_model.train_on_batch([noise2, gen_types2], [positive_y, gen_types2]))
        
    batch_num += 1

