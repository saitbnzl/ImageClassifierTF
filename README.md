# ImageClassifierTF
Simple nueral network for image classification with TensorFlow

#### Usage
To create a new dataset input files should be placed in following format

```
root_of_the_project/data/class1/something.jpg
root_of_the_project/data/class1/something1.jpg
...
root_of_the_project/data/class2/something_else.jpg
root_of_the_project/data/class2/something_else1.jpg
...
```

Then we can create our dataset in csv format using [prepare_data.py](https://github.com/saitbnzl/ImageClassifierTF/blob/master/prepare_data.py):

```
python prepare_data.py
````

After the *dataset.csv* is created run:
```
python SimpleNN.py
````




