import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

rA				= int(sys.argv[1])
rB				= int(sys.argv[2])
r_step			= int(sys.argv[3]) 
cancer_types	= sys.argv[4:]

def plot_data(cancer_type, outfile_prefix, pr_mean, pr_sd, rec_mean, rec_sd, rA, rB, r_step, n_models):
	f1 = plt.figure(figsize = (6,4), dpi=600)

	x_ax_start = r_step
	x_ax_end = rB - rA + r_step
	x_ax = range(x_ax_start, x_ax_end, r_step)
	
	ax1 = plt.subplot(111)
	ax1.plot(x_ax, pr_mean, color='r')
	ax1.fill_between(x_ax,
				pr_mean - pr_sd,
				pr_mean + pr_sd,
				facecolor = 'r',
				alpha = .2 )

	ax1.plot(x_ax, rec_mean, color='b')
	ax1.fill_between(x_ax,
				rec_mean - rec_sd,
				rec_mean + rec_sd,
				facecolor = 'b',
				alpha = .2 )

	plt.title(cancer_type + ' (n = ' + str(n_models) + ')')
	plt.xlabel('# features (starting at ' + str(rA))
	plt.ylabel('Metric (pr = red, rec = blue)')
	plt.grid()
	plt.yticks(np.arange(0.5,1.05,.05))
	plt.xticks(x_ax, rotation = 45)
	f1.savefig(outfile_prefix + '.png')
	plt.close(f1)

for cancer_type in cancer_types:
	f_prefix = cancer_type + '_' + str(rA) + '_' + str(rB) + '_' + str(r_step) + '_min_size_scan'
	prec_fn = f_prefix + '_prec_stats.csv'
	rec_fn = f_prefix + '_rec_stats.csv'
	
	prec_df = pd.read_csv(prec_fn,index_col=0)
	rec_df = pd.read_csv(rec_fn,index_col=0)
	
	pr_mean = np.mean(prec_df.values, axis = 1)
	pr_sd = np.std(prec_df.values, axis = 1)
	
	rec_mean = np.mean(rec_df.values, axis = 1)
	rec_sd = np.std(rec_df.values, axis = 1)
	n_models = prec_df.shape[1]
	
	plot_data(cancer_type, f_prefix, pr_mean, pr_sd, rec_mean, rec_sd, rA, rB, r_step, n_models)
