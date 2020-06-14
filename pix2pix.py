#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:17:33 2019

@author: s4596952
"""
#Importing Libraries


import tensorflow as tf
from tensorflow import keras
import os
import time
#from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset
train_m = np.load('/home/Student/s4596952/pix2pix/Data/train_m.npy')
train_p = np.load('/home/Student/s4596952/pix2pix/Data/train_p.npy')
#test_m = np.load('/data/home/s4596952/Python_projects/CAI_pix2pix/Save_variables/6_pix2pix_data/magnitude_test.ipynb.npy')
#test_p = np.load('/data/home/s4596952/Python_projects/CAI_pix2pix/Save_variables/6_pix2pix_data/phase_test.ipynb.npy')


'''
Building the Generator
    - The Generator is a UNET
    

'''


OUTPUT_CHANNELS = 1

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

#Testing

#down_model = downsample(9, 4)
#down_result = down_model(inp)
#print (down_result.shape)

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


def Generator():
  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 1)

  concat = tf.keras.layers.Concatenate()
  inputs = tf.keras.layers.Input(shape=[None,None,1])   
  #inputs= tf.keras.layers.Input(shape=[None,None,1])
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(x)
  
  model = tf.keras.Model(inputs=inputs, outputs=x)

  return model

generator = Generator()
print(generator.summary())


'''
Build the Discriminator

    The Discriminator is a PatchGAN.
    Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU)
    The shape of the output after the last layer is (batch_size, 30, 30, 1)
    Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called a PatchGAN).
    Discriminator receives 2 inputs.
        Input image (Magnitude Image) and the target image (Actual Phase Image), which it should classify as real.
        Input image and the generated image (output of generator) (Generated Phase Image), which it should classify as fake.
        We concatenate these 2 inputs together in the code (tf.concat([inp, tar], axis=-1))
        
        inp image - magnitude image
        target image - phase image
'''

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 1], name='Magnitude_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 1], name='Phase_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)
  
  last1 = tf.keras.activations.sigmoid(last)
  
  model = tf.keras.Model(inputs=[inp,tar], outputs=last1)
  
  #Compile Model
  discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, loss_weights=[0.5])


  #return model.summary()
  return model 

discriminator = Discriminator()
discriminator.summary()

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = tf.keras.layers.Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = tf.keras.Model(in_src, [dis_out, gen_out])
	# compile model
    
	opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

# define image shape
image_shape = (256,256,1)
# define the models
d_model = discriminator
g_model = generator
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# summarize the model
gan_model.summary()

#Select a batch of random samples, returns images and target
def generate_real_samples(magnitude, phase, n_samples, patch_shape):
    #unpack dataset
    trainA = magnitude
    trainB = phase
    #choose random instances
    i = np.random.randint(0, trainA.shape[0], n_samples)
    #retrive selected images
    X1, X2 = trainA[i], trainB[i]
    #generate 'real' class labels (1)
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y

#generate a batch of images, returns images and targets
def generate_fake_samples(g_model, magnitude, patch_shape):
    #Generate fake instance
    X = g_model.predict(magnitude)
    #Create 'fake' class labels (0)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X,y

#Visit later
X, y = generate_fake_samples(g_model, inp, 30)

X.shape
y.shape

plt.imshow(X[483,:,:,0], cmap='gray')
plt.show()
plt.imshow(y[483,:,:,0], cmap='gray')
plt.show()

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, magnitude, phase, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(magnitude, phase, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	#X_realA = (X_realA + 1) / 2.0
	#X_realB = (X_realB + 1) / 2.0
	#X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + i)
		plt.axis('off')
		plt.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples + i)
		plt.axis('off')
		plt.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples*2 + i)
		plt.axis('off')
		plt.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	plt.savefig(filename1)
	plt.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))


# train pix2pix models
def train(d_model, g_model, gan_model, magnitude, phase, n_epochs=1, n_batch=1):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = magnitude, phase
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(magnitude, phase, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, magnitude, phase)
            


# load image data
#dataset = load_real_samples('maps_256.npz')
print('Loaded', train_m.shape, train_p.shape)
# define input shape based on the loaded dataset
#image_shape = dataset[0].shape[1:]
image_shape = train_m.shape[1:]
# define the models
#d_model = define_discriminator(image_shape)
#g_model = define_generator(image_shape)
d_model = discriminator
g_model = generator
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model, train_m, train_p)



#Saving the final model
g_model.save('pix2pix_generator.h5')




'''
#Predictions

gen_image_p = g_model.predict(test_m)


plt.imshow(test_m[159,:,:,0], cmap='gray')
plt.show()
plt.imshow(test_p[159,:,:,0], cmap='gray')
plt.show()
plt.imshow(gen_image[159,:,:,0], cmap='gray')
plt.show()


#Removing 0-padding
gen_image_p = gen_image_p[:,22:490,224:800,0]

'''
