import numpy as np
import matplotlib.image as img

import os

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

raw_data_path = "./data/"
dirs = os.listdir(raw_data_path)
filenames = []
labels = []
label = 0
for dirc in dirs:
  if not os.path.isfile(dirc):
    files = os.listdir(raw_data_path + dirc)
    for file in files:
      if os.path.isfile(raw_data_path+dirc+"/"+file):
        filenames.append(raw_data_path+dirc+"/"+file)
        labels.append(label)
    label +=1

num_examples = len(filenames)
dataset = np.zeros(num_examples, dtype='int8, object')

index = 0
for file in filenames:
  image = img.imread(file)
  gray = rgb2gray(image)
  gray.resize((100, 100))
  flatten = gray.flatten()
  dataset[index][0] = labels[index]
  dataset[index][1] = ' '.join(map(str, flatten.astype(int)))
  index += 1

with open('dataset.csv', 'wb') as f:
  f.write(b'label,raw_data\n')
  np.savetxt(f, dataset, delimiter=' ',fmt='%i,%s')



