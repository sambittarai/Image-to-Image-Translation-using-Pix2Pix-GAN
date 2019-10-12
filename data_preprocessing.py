
import numpy as np
import cv2
import matplotlib.pyplot as plt

#load the Image
train_m = np.load('train_m.ipynb.npy')
train_p = np.load('train_p.ipynb.npy')
print("Dataset Loaded")

print(train_m.shape)    #(10000,468,576)
print(train_p.shape)    #(10000,468,576)

#Instead of taking random crops, we are cropping the images into 4 images of size (256,256)
#Cropping the original image into 4 images of (256,256)
def crop_image(t_m, t_p):
    y = np.array((30, 286, 30, 286))
    x = np.array((10, 10, 204, 204))
    magnitude = np.zeros((10000*4,256,256))
    cropped_image_m = np.zeros((4,256,256))
    phase = np.zeros((10000*4,256,256))
    cropped_image_p = np.zeros((4,256,256))
    
    for i in range(t_m.shape[0]):
        for j in range(4):
            cropped_image_m[j] = t_m[i, x[j]:x[j]+256, y[j]:y[j]+256]
            cropped_image_p[j] = t_p[i, x[j]:x[j]+256, y[j]:y[j]+256]
        magnitude[i*4:i*4+4,:,:] = cropped_image_m
        phase[i*4:i*4+4,:,:] = cropped_image_p
    return magnitude, phase

#We are doing random cropping only on the training dataset, we will do zero padding on the testing dataset
train_m, train_p = crop_image(train_m, train_p)

print("Magnitude Train Images: ", train_m.shape)     #(40000, 256, 256)
print("Phase Train Images: ", train_p.shape)             #(40000, 256, 256)

#Plotting the training Images
i = 9
print("1")
plt.imshow(m[i,:,:], cmap='gray')
plt.show()
plt.imshow(p[i,:,:], cmap='gray')
plt.show()
print("2")
plt.imshow(magnitude_train[i*4,:,:], cmap='gray')
plt.show()
plt.imshow(phase_train[i*4,:,:], cmap='gray')
plt.show()
print("3")
plt.imshow(magnitude_train[i*4+1,:,:], cmap='gray')
plt.show()
plt.imshow(phase_train[i*4+1,:,:], cmap='gray')
plt.show()
print("4")
plt.imshow(magnitude_train[i*4+2,:,:], cmap='gray')
plt.show()
plt.imshow(phase_train[i*4+2,:,:], cmap='gray')
plt.show()
print("5")
plt.imshow(magnitude_train[i*4+3,:,:], cmap='gray')
plt.show()
plt.imshow(phase_train[i*4+3,:,:], cmap='gray')
plt.show()

#Loading the test Images
test_m = np.load('test_m.ipynb.npy')
test_p = np.load('test_p.ipynb.npy')

print(test_m.shape) # (1200,468,576)
print(test_p.shape) # (1200,468,576)

#The test Images are of uneven shape (468,576), so we will zero pad the images i.e. the final shape will be (512,1024)
#And as our generator is fully convolutional network, we can pass the images directly through the generator

#Zero Pad the test images
m = np.zeros((1200,512,1024))
p = np.zeros((1200,512,1024))

for i in range(1200):
    m[i,22:490,224:800] = test_m[i,:,:]
    p[i,22:490,224:800] = test_p[i,:,:]
    
test_m = m
test_p = p
print("magnitude_test: ", test_m.shape) # (1200,512,1024)
print("phase_test: ", test_p.shape) # (1200,512,1024)



#Assign dtype as float32
train_m = train_m.astype('float32')
train_p = train_p.astype('float32')
test_m = test_m.astype('float32')
test_p = test_p.astype('float32')

#Normalizing the data b/w (-1,1)
train_m = (2*train_m/np.max(train_m)) - 1
train_p = (train_p/np.max(train_p))
test_m = (2*test_m/np.max(test_m)) - 1
test_p = (test_p/np.max(test_p))

#Adding an extra dimension
train_m = np.expand_dims(train_m, axis=3)
train_p = np.expand_dims(train_p, axis=3)
test_m = np.expand_dims(test_m, axis=3)
test_p = np.expand_dims(test_p, axis=3)


#Saving the final data

#Saving the training images for pix2pix of size (40000,256,256,1)
np.save('/data/home/s4596952/Python_projects/CAI_pix2pix/Save_variables/6_pix2pix_data/magnitude_train.ipynb', magnitude_train)
np.save('/data/home/s4596952/Python_projects/CAI_pix2pix/Save_variables/6_pix2pix_data/phase_train.ipynb', phase_train)


#Saving the test images for pix2pix of size (1200,512,1024,1)
np.save('/data/home/s4596952/Python_projects/CAI_pix2pix/Save_variables/6_pix2pix_data/magnitude_test.ipynb', magnitude_test)
np.save('/data/home/s4596952/Python_projects/CAI_pix2pix/Save_variables/6_pix2pix_data/phase_test.ipynb', phase_test)


print("Done")


































































