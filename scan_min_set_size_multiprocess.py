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

# extra functionality
import ml_utils

# multithreading
import multiprocessing as mp

# for cleanup
import gc

# this is supposed to prevent keras / tf from taking all the gpu memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# change this as necessary
rank_files_suffix = '_fractional_full_imp_sorted.csv'

data_file	= sys.argv[1]			
rangeA		= int(sys.argv[2])
rangeB		= int(sys.argv[3])
range_step	= int(sys.argv[4])
n_models	= int(sys.argv[5])
N_epochs	= int(sys.argv[6])
cancer_types	= sys.argv[7:]

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

Ys = {}
Y = np.zeros((data.shape[0],),dtype=int)
for cancer_type in cancer_types:
	Ys[cancer_type] = Y.copy()
	Ys[cancer_type][data['ONCOTREE_CODE'] == cancer_type] = 1
	print("%s samples:\t%d" % (cancer_type, np.sum(Ys[cancer_type],axis=0)) )

def get_min_set_size(	cancer_type,
						X,
						Y,
						rangeA,
						rangeB,
						range_step,
						N_epochs,
						n_models):
							
	# setup tf, see above
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	set_session(sess)
	
	sorted_imp_file = cancer_type + rank_files_suffix
	print("Loading ranked features from " + sorted_imp_file)
	
	sorted_imp_full = pd.read_csv(sorted_imp_file, delimiter=',', index_col=0,header=None)
	
	scan_range = range(rangeA, rangeB, range_step)
	
	prec_scores_0		= np.zeros( ( len(scan_range), len(range(n_models) ) ) )
	recall_scores_0		= np.zeros( ( len(scan_range), len(range(n_models) ) ) )

	prec_scores_1		= np.zeros( ( len(scan_range), len(range(n_models) ) ) )
	recall_scores_1		= np.zeros( ( len(scan_range), len(range(n_models) ) ) )

	outfile_prefix = cancer_type +  '_' + str(rangeA) + '_' + str(rangeB) + '_' + str(range_step) + '_min_size_scan'
	
	class_w = {0: 1, 1: float(X.shape[0])/np.sum(Y,axis=0)}

	for bin_counter in range(len(scan_range)):
		l1 = list(range(rangeA, rangeA + (bin_counter+1)*range_step))
		Xm = X.loc[:,sorted_imp_full.index[l1].map(str)]
		# print(Xm.shape)
		
		for m in range(n_models):
			# split data
			x, x_te, y, y_te = train_test_split(Xm.values, Y, test_size = 0.2)
		
			# build a model
			model = ml_utils.build_model(
									r_lambda=.025,
									input_dim = x.shape[1],
									layers=[4],
									activations=['sigmoid'])
									
			model.fit(	x,
						y,
						epochs = N_epochs,
						batch_size=32,
						class_weight=class_w,
						shuffle=True,
						verbose = 0)
			
			y_pred = model.predict_classes(x_te)
			cl_report = classification_report(y_te,y_pred, output_dict = True)
			
			prec_scores_0[bin_counter,m]	= cl_report['0']['precision']
			recall_scores_0[bin_counter,m]	= cl_report['0']['recall']
	
			prec_scores_1[bin_counter,m] 	= cl_report['1']['precision']
			recall_scores_1[bin_counter,m]	= cl_report['1']['recall']
		
			prec_df = pd.DataFrame(data=prec_scores_1)
			prec_df.to_csv(outfile_prefix + '_prec_stats.csv')

			rec_df = pd.DataFrame(data=recall_scores_1)
			rec_df.to_csv(outfile_prefix + '_rec_stats.csv')
			
			print("%s\t%d\t%d\t%d\t%.5f\t%.5f" % (
											cancer_type,
											m,
											rangeA,
											rangeA + (bin_counter+1)*range_step,
											prec_scores_1[bin_counter,m],
											recall_scores_1[bin_counter,m]))
											
			# free memory
			del model
			del x
			del x_te
			del y
			del y_te
			gc.collect()
		
		del Xm
		gc.collect()

def runInParallel(	cancer_types = None,
					X = None,
					Ys = None,
					rangeA = None,
					rangeB = None,
					range_step = None,
					N_epochs = None,
					n_models = None):
	proc = []
	for cancer_type in cancer_types:
		p = mp.Process(target=get_min_set_size, args=(
													cancer_type,
													X,
													Ys[cancer_type],
													rangeA,
													rangeB,
													range_step,
													N_epochs,
													n_models))
		p.start()
		proc.append(p)
	
	for p in proc:
		p.join()

runInParallel(	cancer_types = cancer_types,
				X = X,
				Ys = Ys,
				rangeA = rangeA,
				rangeB = rangeB,
				range_step = range_step,
				N_epochs = N_epochs,
				n_models = n_models)
