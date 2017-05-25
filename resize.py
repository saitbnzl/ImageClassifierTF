from PIL import Image
import os, sys

path = "./data/"
dirs = os.listdir( path )
def resize():
    for dirc in dirs:
        files = os.listdir( path + dirc )
        for file in files:
            print(path+dirc+"/"+file)
            if os.path.isfile(path+dirc+"/"+file):
            	im = Image.open(path+dirc+"/"+file)
            	f, e = os.path.splitext(path+dirc+"/"+file)
            	imResize = im.resize((100,100), Image.ANTIALIAS)
            	imResize.save(f, 'JPEG', quality=90)
            	
resize()
