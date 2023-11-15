#show_pkl.py
 
import pickle
path1 = 'data/Transient/Transient_Attributes_word2vec_29.pkl'   #path='/root/……/aus_openface.pkl'   pkl文件所在路径

path2 = 'data/coco/coco_glove_word2vec.pkl'
	   
f = open(path1,'rb')
data1 = pickle.load(f)

print(type(data1[0]))

f = open(path2,'rb')
data2 = pickle.load(f)

print(type(data2[0]))