# ver 20170521 by jian: use vgg16 in keras to infer the bottleneck features of selected dogs-vs-cats images
# lesson learnt: to run vgg inference, the image can be resized differently

# lsf command: bsub -q gputest -R "rusage[ngpus_excl_p=1]" -Is tcsh

import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import os

files = os.listdir('../input/train/')
model = VGG16(include_top=False)

def inf(f,size=224):
	img = image.load_img('../input/train/'+f, target_size=(size,size))
	x= image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	features = model.predict(x)
	return features

o = inf(files[2])
print(o.shape)
#(1, 7, 7, 512)
o = inf(files[20])
print(o.shape)
#(1, 7, 7, 512)
o = inf(files[20],size=128)
print(o.shape)
#(1, 4, 4, 512)
o = inf(files[200])
print(o.shape)
#(1, 7, 7, 512)
o = inf(files[200],size=512)
print(o.shape)
#(1, 16, 16, 512)

