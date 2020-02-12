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
#from sklearn.metrics import precision_recall_curve, precision_score, recall_score

# extra functionality
import ml_utils

# for timing
import time

# multithreading
import multiprocessing as mp

# this is supposed to prevent keras / tf from taking all the gpu memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True	# dynamically grow the memory used on the GPU
#config.log_device_placement = True 	# to log device placement (on which device the operation ran)
										# nothing gets printed in Jupyter, only if you run it standalone)
#sess = tf.Session(config=config)
#set_session(sess)						# set this TensorFlow session as the default session for Keras

# for cleanup
import gc

data_file		= sys.argv[1] 
e_start			= int(sys.argv[2])
e_end			= int(sys.argv[3])
e_step			= int(sys.argv[4])
n_models		= int(sys.argv[5])

cancer_types	= sys.argv[6:]
print(cancer_types)

e_range = range(e_start, e_end + e_step, e_step)

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

X = data.iloc[:,4:]
X[X > 10] = 10

Ys = {}
Y = np.zeros((data.shape[0],),dtype=int)
for cancer_type in cancer_types:
	Ys[cancer_type] = Y.copy()
	Ys[cancer_type][data['ONCOTREE_CODE'] == cancer_type] = 1
	print("%s samples:\t%d" % (cancer_type, np.sum(Ys[cancer_type],axis=0)) )

# free memory
del data
gc.collect()

def get_N_epochs(cancer_type, X, Y, e_range, n_models):
	# setup tf, see above
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	set_session(sess)
	
	# init stats arrays
	prec_scores_0		= np.zeros( ( len(e_range), len(range(n_models) ) ) )
	recall_scores_0		= np.zeros( ( len(e_range), len(range(n_models) ) ) )

	prec_scores_1		= np.zeros( ( len(e_range), len(range(n_models) ) ) )
	recall_scores_1		= np.zeros( ( len(e_range), len(range(n_models) ) ) )

	outfile_prefix = cancer_type +  '_' + str(e_start) + '_' + str(e_end) + '_' + str(e_step)

	class_w = {0: 1, 1: float(X.shape[0])/np.sum(Y,axis=0)}

	for m in range(n_models):
		last_epoch = 0
		
		# split data
		x, x_te, y, y_te = train_test_split(X.values, Y, test_size = 0.2)
		
		# build a model
		model = ml_utils.build_model(r_lambda=.025,
									input_dim=X.shape[1],
									layers=[4],
									activations=['sigmoid'])
		
		for E in range(len(e_range)):
			print("%s\t%d\t%d\t%d" % (cancer_type, m, last_epoch, e_range[E]), end = '\t')
			
			model.fit(x,y,
						epochs = e_range[E],
						batch_size=32,
						class_weight=class_w,
						shuffle=True,
						verbose = 0,
						initial_epoch = last_epoch)
			
			y_pred = model.predict_classes(x_te)
			cl_report = classification_report(y_te,y_pred, output_dict = True)
			
			prec_scores_0[E,m] = cl_report['0']['precision']
			recall_scores_0[E,m] = cl_report['0']['recall']

			prec_scores_1[E,m] = cl_report['1']['precision']
			recall_scores_1[E,m] = cl_report['1']['recall']
			
			# output stats to screen
			print ("%.5f\t%.5f\t%.5f\t%.5f" %(prec_scores_0[E,m],recall_scores_0[E,m], prec_scores_1[E,m],recall_scores_1[E,m]) )
		
			# write partial results
			prec_df = pd.DataFrame(data=prec_scores_1)
			prec_df.to_csv(outfile_prefix + '_prec_stats.csv')

			rec_df = pd.DataFrame(data=recall_scores_1)
			rec_df.to_csv(outfile_prefix + '_rec_stats.csv')
			
			last_epoch = e_range[E]

		# free memory
		del model
		del x
		del x_te
		del y
		del y_te
		gc.collect()
	
	#prec_mean = np.mean(prec_scores_1, axis = 1)
	#rec_mean = np.mean(recall_scores_1, axis = 1)

	#prec_sd = np.std(prec_scores_1, axis = 1)
	#rec_sd = np.std(recall_scores_1, axis = 1)

	## plot results and save
	#f1 = plt.figure(figsize = (6,4), dpi=120)

	#plt.plot(e_range, prec_mean, color='r')
	#plt.fill_between(e_range,
				#prec_mean - prec_sd,
				#prec_mean + prec_sd,
				#facecolor = 'r',
				#alpha = .2 )

	#plt.plot(e_range, rec_mean, color='b')
	#plt.fill_between(e_range,
				#rec_mean - rec_sd,
				#rec_mean + rec_sd,
				#facecolor = 'b',
				#alpha = .2 )
	#plt.xlabel("N epochs")
	#plt.ylabel('Metric (pr = red, rec = blue)')
	#plt.title(cancer_type + ' (n = ' + str(n_models) + ')')
	#plt.grid()
	#plt.yticks(np.arange(0.8,1.025,.025))
	#plt.xticks(e_range, rotation='vertical')
	#f1.savefig(outfile_prefix + '_N_epoch_scan.png')
	#plt.close(f1)
	
def runInParallel(cancer_types = None, X = None, Ys = None, e_range = None, n_models = None):
	proc = []
	for cancer_type in cancer_types:
		p = mp.Process(target=get_N_epochs, args=(cancer_type, X, Ys[cancer_type], e_range, n_models))
		p.start()
		proc.append(p)
	for p in proc:
		p.join()

runInParallel(cancer_types = cancer_types, X = X, Ys = Ys, e_range = e_range, n_models = n_models)
