# general imports
import sys
import pandas as pd
import numpy as np

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

# extra functionality
import ml_utils

# for cleanup
import gc

# multithreading
import multiprocessing as mp

# this is supposed to prevent keras / tf from taking all the gpu memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def parse_c_list(c_list):
	c_dict = {}
	for item in c_list:
		l = item.split(':')
		c_dict[l[0]] = {}
		c_dict[l[0]]['f_num'] = int(l[1])
		c_dict[l[0]]['nE'] = int(l[2])
	
	return c_dict

data_file	= sys.argv[1] 
id_map_file	= sys.argv[2]
n_models	= int(sys.argv[3])
fractional	= sys.argv[4].lower() == 'true'
c_list		= sys.argv[5:]

# change this as needed
infile_suffix = '_fractional_full_imp_sorted.csv'

print('Loading ID mapping from ' + id_map_file)
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

print("Loading data from " + data_file)
data = pd.read_csv(data_file, delimiter='\t', names=header, header=1,dtype=dtypes)

X = data.iloc[:,4:]
X[X > 10] = 10

c_dict = parse_c_list(c_list)
print(c_dict)

Ys = {}
Y = np.zeros((data.shape[0],),dtype=int)
for cancer_type in c_dict.keys():
	Ys[cancer_type] = Y.copy()
	Ys[cancer_type][data['ONCOTREE_CODE'] == cancer_type] = 1
	print("%s samples:\t%d" % (cancer_type, np.sum(Ys[cancer_type],axis=0)) )

# free memory
del data
gc.collect()

def get_min_perm_importance(cancer_type, X, Y, c_dict, n_models):
	# setup tf, see above
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	set_session(sess)
	
	sorted_imp_file = cancer_type + infile_suffix
	print("Loading ranked features from " + sorted_imp_file)
	sorted_imp_full = pd.read_csv(sorted_imp_file, delimiter=',', index_col=0, header=None)
	
	if fractional:
		outfile_prefix = cancer_type + '_fractional_min_imp'
	else:
		outfile_prefix = cancer_type + '_min_imp'
	
	f_num = c_dict[cancer_type]['f_num']
	l1 = list(range(f_num))
	Xm = X.loc[:,sorted_imp_full.index[l1].map(str)]
	
	print("%d\t%d\t%d" % (Xm.shape[1], n_models, c_dict[cancer_type]['nE']))
	
	imp = ml_utils.iter_perm_importance(X = Xm,
										Y = Y,
										feat_names = sorted_imp_full.index[range(f_num)],
										n_iter = n_models,
										N_epochs = c_dict[cancer_type]['nE'],
										verbose = True,
										fractional = fractional)

	print( "%s\t%.5f\t%.5f\n" %(cancer_type,
								np.mean(imp['pr']),
								np.mean(imp['rec'])) )

	imp['fi_df'].to_csv(outfile_prefix + '.csv',header=False)
	
	sorted_imp = pd.Series(index=imp['fi_df'].index, data=np.mean(imp['fi_df'].values, axis=1))
	sorted_imp.sort_values(ascending=False,inplace=True)
	sorted_imp.to_csv(outfile_prefix + '_sorted.csv', header=False)
	
	index2gene = [id_map.loc[id_map['id'] == fn,'name'].values[0] for fn in imp['fi_df'].index.map(str)]
	sorted_imp_names = pd.Series(index=index2gene, data=np.mean(imp['fi_df'].values, axis=1))
	sorted_imp_names.sort_values(ascending=False,inplace=True)
	sorted_imp_names.to_csv(outfile_prefix + '_sorted_names.tsv',header=False, sep='\t')

	prec_df = pd.DataFrame(data=imp['pr'])
	prec_df.to_csv(outfile_prefix + '_prec_stats.csv')

	rec_df = pd.DataFrame(data=imp['rec'])
	rec_df.to_csv(outfile_prefix + '_rec_stats.csv')
	
	del Xm
	gc.collect()

def runInParallel(X = None, Ys = None, c_dict = None, n_models = None):
	proc = []
	for cancer_type in c_dict.keys():
		p = mp.Process(target=get_min_perm_importance, 
						args=(	cancer_type,
								X,
								Ys[cancer_type],
								c_dict,
								n_models))
		p.start()
		proc.append(p)
		
	for p in proc:
		p.join()

runInParallel(X = X, Ys = Ys, c_dict = c_dict, n_models = n_models)
