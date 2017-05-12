# ver 20170511 by jian: use Xception as pretrained model to play dogs-vs-cats competition on Kaggle
'''
ref
Original competition: https://www.kaggle.com/c/dogs-vs-cats
``modern version of this competition'': https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
similar/same idea as in http://fastml.com/classifying-images-with-a-pre-trained-deep-network/
cat terms: https://github.com/zygmuntz/kaggle-cats-and-dogs/blob/master/overfeat/data/cats.txt
dog terms: https://github.com/zygmuntz/kaggle-cats-and-dogs/blob/master/overfeat/data/dogs.txt


quick lesson:
- even this ``simple'' task is very time comsuming
'''

import numpy as np
import os
import pandas as pd

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image

# In case you use tensorflow 1.1.0, you may use tensorflow.contrib.keras (not fully tested though)


model = Xception()
files = os.listdir('../input/train') # Assert the folder structure

def pred (m,n):
	output = pd.DataFrame(columns=['id','class','prob','truth','file'], index=None)
	for j in range(m,n):
		file = files[j]
		img = image.load_img('../input/train/'+file,target_size=(299,299))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		preds = model.predict(x)
		imagenet_class = decode_predictions(preds)[0][0]
		output = output.append({'id':imagenet_class[0],'class':imagenet_class[1],'prob':imagenet_class[2],'truth': file.split('.')[0],'file':file},ignore_index=True)
	return output

pred(0,len(files)).to_csv('raw_prediction.csv')
#pred(0,55).to_csv('raw_prediction.csv')
