# ver 20170511 by jian : 
import pandas as pd
raw = pd.read_csv('raw_prediction.csv')
cf = open('kaggle-cats-and-dogs\overfeat\data\cats.txt')
df = open('kaggle-cats-and-dogs\overfeat\data\dogs.txt')
cats = cf.read()
cats = cats.split( "\n" )
dogs = df.read()
dogs = dogs.split( "\n" )

dog_labels = [x.split(',')[0] for x in dogs]
cat_labels = [x.split(',')[0] for x in cats]

raw['pred'] = ''
for r in range(raw.shape[0]):
    cls = raw['class'][r].replace('_',' ')
    if cls in dog_labels:
        raw['pred'][r] = 'dog'
    elif cls in cat_labels:
        raw['pred'][r] = 'cat'
print(raw[raw['truth'] != raw['pred']])

quit()
raw['id2'] = raw.id.apply(lambda x: '{0:b}'.format(int(x[1:9])))
labels = raw.groupby(['class','truth','id','id2']).size()
labels.to_csv('imagenet_label_to_dogs_and_cats.csv')
