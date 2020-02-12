# general imports
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
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

# for timing
import time

# for cleanup
import gc

def build_model(r_lambda=0,input_dim=None,layers=None, activations=None):
	'''builds a keras model '''
	model = Sequential()
	model.add(Dense(layers[0],
					activation=activations[0],
					input_dim = input_dim,
					kernel_regularizer=regularizers.l2(r_lambda) ))

	for i in range(1,len(layers)):
		model.add(Dense(layers[i],
						activation=activations[i],
						kernel_regularizer=regularizers.l2(r_lambda) ))

	model.add(Dense(1, activation = 'sigmoid'))
	opt = Adagrad()

	model.compile(loss='binary_crossentropy', 
					optimizer=opt,
					metrics=['accuracy'])

	#model.summary()
	return model


def plot_training(model):
	''' plots results of training a keras model '''
	
	fig = plt.figure(figsize=(12,6),dpi=120)
	ax1 = fig.add_subplot(121)
	ax1.plot(model.history.epoch,
			model.history.history['acc'], 
			'r-',
			label='Train Acc')

	ax1.plot(model.history.epoch,
			model.history.history['val_acc'],
			'b-',
			label='Val Acc')

	plt.title('Accuracy')
	plt.ylim((0.5,1))
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	ax1.legend()

	ax2 = fig.add_subplot(122)
	ax2.plot(model.history.epoch,
			model.history.history['loss'],
			'b-',
			label='Train Loss')

	ax2.plot(model.history.epoch,
			model.history.history['val_loss'],
			'g-',
			label='Val Loss')

	plt.title('Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	ax2.legend()

	return fig


def plot_prec_recall_curves(model, x_te, y_te):
	''' plots precission-recall curve '''
	y_score = model.predict_proba(x_te)

	precision, recall, _ = precision_recall_curve(y_te, y_score )

	fig = plt.figure(figsize=(6,6),dpi=120)

	ax = fig.add_subplot(111)
	ax.plot(recall, precision, color='b')
	ax.legend()
	plt.xlabel('Recall')
	plt.ylabel('Precission')
	plt.ylim(.5,1)

	return fig


def perm_importance(model = None, X = None, perm_space = None, n_iter = 5, verbose = False, fractional = False):
	''' performs permutation importance feature ranking '''
	
	# compute probabilities
	full_proba = np.tile(model.predict_proba(X), X.shape[1])
	
	# Z will be used in the loop below
	# the permuted features are replenished from the original data
	Z = X.copy()						# !cleanup!
	pert_proba = np.zeros(X.shape)		# !cleanup!
	n_feat = X.shape[1]
	
	# permute features
	for f in range(n_feat):
		tmp_proba = np.zeros((X.shape[0], n_iter))	#!cleanup!
		for it in range(n_iter):
			if verbose:
				print("[%d / %d / %d]" %(it, f, n_feat), end='\r')

			# create permuted feature drawing from the same distribution
			noise_feature = np.random.choice(perm_space[:,f], size = X.shape[0], replace = False)
			Z[:,f] = noise_feature
			tmp_proba[:,it] = model.predict_proba(Z).flatten()

		# take the average of the effects
		pert_proba[:,f] = np.mean(tmp_proba, axis=1)
		
		# recover the original feature before moving on
		Z[:,f] = X[:,f]
	
	if fractional:
		cont_means = np.mean((full_proba - pert_proba)/full_proba, axis=0)
		cont_sds = np.std((full_proba - pert_proba)/full_proba, axis=0)
	
	else:
		cont_means = np.mean(full_proba - pert_proba, axis=0)
		cont_sds = np.std(full_proba - pert_proba, axis=0)
	
	# free memory
	del Z
	del pert_proba
	del tmp_proba
	del noise_feature
	gc.collect()
	
	return [cont_means, cont_sds]

def plot_feature_importances(name = None, imps = None, xticks_spacing = 250): 
	''' plots ranked feature importances '''
	fig = plt.figure(figsize=(20,18),dpi=600)
	fig.suptitle(name + ' (n = ' + str(imps.shape[1]) + ')')
	imps_stats = pd.DataFrame(index = imps.index)
	imps_stats['mean'] = np.mean(imps.iloc[:,1:].values, axis = 1)
	imps_stats['std'] = np.std(imps.iloc[:,1:].values, axis = 1)
	imps_stats.sort_values(by=['mean'],ascending=False,inplace=True)
	x_data = np.arange(imps.shape[0])

	ax1 = plt.subplot(411)
	ax1.fill_between(x_data,
					imps_stats['mean'] - imps_stats['std'],
					imps_stats['mean'] + imps_stats['std'],
					facecolor='#00FF00', # green
					alpha = 1)

	plt.xticks(np.arange(0,imps.shape[0],xticks_spacing),rotation='vertical')
	plt.ylabel('Mean importance')
	plt.xlabel('Rank#')
	plt.grid()

	ax2 = plt.subplot(412)
	ax2.fill_between(x_data, 
					np.zeros(imps_stats['mean'].shape[0]), 
					np.abs(imps_stats['mean']/imps_stats['std']), 
					alpha = .5,
					facecolor = 'r')
	#ax2.plot(x_data,
	#		np.ma.average(np.abs(imps_stats['std']/imps_stats['mean']),axis=1),
	#		color = 'k')
					
	plt.xticks(np.arange(0,imps.shape[0],xticks_spacing),rotation='vertical')
	plt.ylabel('abs( mean / std)')
	plt.xlabel('Rank#')
	plt.ylim(0,5)
	plt.yticks(np.arange(0,5,.25))
	plt.grid()
	
	ax3 = plt.subplot(413)
	ax3.plot(x_data, np.cumsum(imps_stats['mean'])/np.sum(np.abs(imps_stats['mean'])))
	plt.xticks(np.arange(0,imps.shape[0],xticks_spacing),rotation='vertical')
	plt.ylabel('Fraction of cumulative sum')
	plt.xlabel('Rank#')
	plt.grid()

	ax4 = plt.subplot(414)
	ax4.plot(x_data, np.gradient(np.cumsum(imps_stats['mean'])))
	plt.xticks(np.arange(0,imps.shape[0],xticks_spacing),rotation='vertical')
	plt.ylabel('Gradient Cumulative Sum')
	plt.xlabel('Rank#')
	plt.grid()

	return fig
	
def train_model(X = None, Y = None,
				N_epochs = None, r_lambda = .025, prefix = None, iter_num = None,
				save_fig = False, plot = False, make_figs = False, verbose = True,
				initial_epoch = 0):
					
	''' trains a keras model and plots a bunch of stuff if asked to '''
	start_time = time.time()
	
	class_w = {0: 1, 1: float(X.shape[0])/np.sum(Y,axis=0)}
	#print(class_w)

    # split data
	x, x_te, y, y_te = train_test_split(X.values, Y, test_size = 0.2)

	#build model
	model = build_model(r_lambda=.025,
					input_dim=X.shape[1],
					layers=[4],
					activations=['sigmoid'])

	# train
	history = model.fit(x,y,
					epochs=N_epochs,
					batch_size=32,
					class_weight=class_w,
					validation_split=.1,
					shuffle=True,
					verbose = 0,
					initial_epoch = initial_epoch)

	y_pred = model.predict_classes(x_te)
	cl_report = classification_report(y_te,y_pred, output_dict = True)
	
	prec_score_0 = cl_report['0']['precision']
	prec_score_1 = cl_report['1']['precision']
		
	recall_score_0 = cl_report['0']['recall']
	recall_score_1 = cl_report['1']['recall']
	
	if verbose:
		print ("%.5f\t%.5f\t%.5f\t%.5f" %(prec_score_0,recall_score_0, prec_score_1,recall_score_1), end = '\t' )
	
	f1 = None
	f2 = None
	f3 = None
	
	if make_figs:
		f1 = plot_training(model)
		f2 = plot_prec_recall_curves(model,x_te,y_te)
		f3 = plot_class_separation(model = model, X = x_te, Y = y_te, ylim = (0, .05))
	
	if plot:
		f1.show()
		f2.show()
		f3.show()
	
	if save_fig:  
		f1.savefig(prefix + str(iter_number) + '_training.png')
		f2.savefig(prefix + str(iter_number) + '_pr_curve.png')
		f3.savefig(prefix + str(iter_number) + '_pr_csep.png')
	
	end_time = time.time()
	if verbose:
		print ("%.4f" % (end_time-start_time))
	
	return {'model' : model,
            'history' : history,
            'x_train' : x,
            'x_test' : x_te,
            'y_train' : y,
            'y_test' : y_te,
            'class_weights' : class_w,
            "figs" : [f1, f2, f3],
            "report" : cl_report}

def plot_class_separation(model = None, X = None, Y = None, ylim = (0,1)):
	''' plots class separetion of a predictor '''
	proba = model.predict_proba(X).flatten()
	fig = plt.figure(figsize=(8,4))
	ax = plt.subplot(111)
	ax.bar(x = np.arange(0,1,.01), 
			height = np.histogram(proba[Y==0], bins = np.arange(0,1.01,.01))[0]/np.sum(Y==0),
			width = .01,
			alpha = .5)

	ax.bar(x = np.arange(0,1,.01),
			height = np.histogram(proba[Y==1], bins = np.arange(0,1.01,.01))[0]/np.sum(Y==1),
			width = .01,
			alpha = .5)

	plt.ylim(ylim)
	plt.xlabel('Proba')
	plt.ylabel('fraction of samples')
	plt.xticks(np.arange(0,1.1,.1))

	return fig

def iter_perm_importance(X = None, Y = None, n_iter=5, feat_names = None, 
						N_epochs = 500, perm_iter = 10, fractional = False, verbose = False):
	''' computes feature importance from a number of different models for better statistics '''
	iter_range = range(n_iter)
	fi_df = pd.DataFrame(index=feat_names)
	
	prec_scores_0 = np.zeros(n_iter)
	recall_scores_0 = np.zeros(n_iter)
	
	prec_scores_1 = np.zeros(n_iter)
	recall_scores_1 = np.zeros(n_iter)
	
	class_w = {0: 1, 1: float(X.shape[0])/np.sum(Y,axis=0)}
	
	for i in iter_range:
		# split data, !cleanup!
		x, x_te, y, y_te = train_test_split(X.values, Y, test_size = 0.2)
		
		# build a model, !cleanup!
		model = build_model(r_lambda=.025,
							input_dim=X.shape[1],
							layers=[4],
							activations=['sigmoid'])
		
		# train the model
		model.fit(x,y,
					epochs = N_epochs,
					batch_size=32,
					class_weight=class_w,
					shuffle=True,
					verbose = 0)
					
		y_pred = model.predict_classes(x_te)
		cl_report = classification_report(y_te,y_pred, output_dict = True)
			
		prec_scores_0[i]	= cl_report['0']['precision']
		recall_scores_0[i] = cl_report['0']['recall']

		prec_scores_1[i]	= cl_report['1']['precision']
		recall_scores_1[i] = cl_report['1']['recall']

		if verbose:
			print ("%d\t%.5f\t%.5f\t%.5f\t%.5f" %(i,prec_scores_0[i],
													recall_scores_0[i],
													prec_scores_1[i],
													recall_scores_1[i]) )					

		fi = perm_importance(model = model,
							X = X.values[Y==1,:], 
							perm_space = X.values[Y==0,:], 
							n_iter = perm_iter,
							verbose = verbose,
							fractional = fractional)

		# free memory					
		del model
		del x_te
		del y_te
		del x
		del y
		gc.collect()

		fi_df['fi_'+ str(i)] = fi[0]

	return {'fi_df' : fi_df,
			'pr' : prec_scores_1,
			'rec' : recall_scores_1}

def plot_diff_expression(X = None, Y = None, imps = None, fiA = None, fiB = None):
	fig = plt.figure(figsize = (6,4), dpi=120)
	
	plt.hist([np.mean(X.values[Y==0,X.columns.get_loc(str(i))],axis=0) for i in imps.index[fiA:fiB].map(str)], alpha = .5, bins = np.arange(-2,2, .1))
	plt.hist([np.mean(X.values[Y==1,X.columns.get_loc(str(i))],axis=0) for i in imps.index[fiA:fiB].map(str)], alpha = .5, bins = np.arange(-2,2, .1))
	
	plt.xlabel('Expression')
	plt.ylabel('# genes')
	return fig
