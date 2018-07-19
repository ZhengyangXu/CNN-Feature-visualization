
# coding: utf-8

# In[1]:


from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

import matplotlib.pyplot as plt
from pylab import *

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

img_path = ".\\data\\1.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
feature=block4_pool_features
#block4_pool_features.shape (1, 14, 14, 512)

#from PIL import Image
#feature1=feature.reshape(196,512)
#new_im = Image.fromarray(feature1)
#new_im.show()

#draw and save pic
for i in range(10):
    plt.imshow(feature[0,:,:,i])
    #plt.savefig('./data/ck'+str(i),dpi=60)
    show()


# In[ ]:




