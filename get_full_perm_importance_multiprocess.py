# general imports
import sys
import pandas as pd
import numpy as np
import string

# keras imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adagrad
from keras import regularizers

# for plotting
import matplotlib.pyplot as plt

# for splitting data
from sklearn.model_selection import train_test_split

# for reporting
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

# extra functionality
import ml_utils

# for cleanup
import gc

# multithreading
import multiprocessing as mp

# this is supposed to prevent keras / tf from taking all the gpu memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def parse_cancer_epochs(c_list):
	c_dict = {}
	for item in c_list:
		l = item.split(':')
		c_dict[l[0]] = int(l[1])
	
	return c_dict

data_file	= sys.argv[1] 
id_map_file	= sys.argv[2]
n_models	= int(sys.argv[3])
fractional	= sys.argv[4].lower() == 'true'
c_list		= sys.argv[5:]

c_dict = parse_cancer_epochs(c_list)
print(c_dict)

print('Loading ID mapping...')
id_map = pd.read_csv(id_map_file, delimiter='\t',names=['name','id'])

h = open(data_file)
header = h.readline().split("\t")
header = [label.rstrip("\n") for label in header]
h.close()

dtypes = {}
for label in header:
	if label[0].isdigit():
		dtypes[label] = 'float32'
	else:
		dtypes[label] = 'object'

print("Loading data...")
data = pd.read_csv(data_file, delimiter='\t', names=header, header=1,dtype=dtypes)
feat_names = data.columns.values.tolist()[4:]
feat_names = [f.rstrip("\n") for f in feat_names]

X = data.iloc[:,4:]
X[X > 10] = 10

Ys = {}
Y = np.zeros((data.shape[0],),dtype=int)
for cancer_type in c_dict.keys():
	Ys[cancer_type] = Y.copy()
	Ys[cancer_type][data['ONCOTREE_CODE'] == cancer_type] = 1
	print("%s samples:\t%d" % (cancer_type, np.sum(Ys[cancer_type],axis=0)) )

# free memory
del data
gc.collect()

def get_full_perm_importance(cancer_type, X, Y, N_epochs, n_models,feat_names):
	# setup tf, see above
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	set_session(sess)
	
	if fractional:
		outfile_prefix = cancer_type + '_fractional_full_imp'
	else:
		outfile_prefix = cancer_type + '_full_imp'
	
	imp_full = ml_utils.iter_perm_importance(X = X,
											Y = Y,
											N_epochs = N_epochs,
											n_iter = n_models,
											perm_iter = 10,
											feat_names=feat_names,
											verbose = True)

	print( "%s\t%.5f\t%.5f\n" %(cancer_type,np.mean(imp_full['pr']), np.mean(imp_full['rec'])) )

	imp_full['fi_df'].to_csv(outfile_prefix + '.csv',header=False)
	sorted_imp_full = pd.Series(index=imp_full['fi_df'].index, data=np.mean(imp_full['fi_df'].values, axis=1))
	sorted_imp_full.sort_values(ascending=False,inplace=True)
	sorted_imp_full.to_csv(outfile_prefix + '_sorted.csv', header=False)
	
	prec_df = pd.DataFrame(data=imp_full['pr'])
	prec_df.to_csv(outfile_prefix + '_prec_stats.csv')

	rec_df = pd.DataFrame(data=imp_full['rec'])
	rec_df.to_csv(outfile_prefix + '_rec_stats.csv')

def runInParallel(X = None, Ys = None, c_dict = None, n_models = None, feat_names= None):
	proc = []
	for cancer_type in c_dict.keys():
		p = mp.Process(target=get_full_perm_importance, 
						args=(cancer_type, X, Ys[cancer_type], c_dict[cancer_type], n_models, feat_names))
		p.start()
		proc.append(p)
	for p in proc:
		p.join()

runInParallel(X = X, Ys = Ys, c_dict = c_dict, n_models = n_models, feat_names = feat_names)
