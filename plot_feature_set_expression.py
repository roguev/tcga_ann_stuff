import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_file	= sys.argv[1]
id_map_file	= sys.argv[2]
fn_middle	= sys.argv[3]
c_list		= sys.argv[4:]

c_dict = {}
for item in c_list:
	l = item.split(':')
	c_dict[l[0]] = {}
	c_dict[l[0]]['A'] = int(l[1])
	c_dict[l[0]]['B'] = int(l[2])

def do_plotting(cancer_type = None, X = None, 
				Ys = None, c_dict = None, imp = None,
				oe_thr = .5, ue_thr = -.5, window_step = 10):
	
	print(cancer_type)
	
	A = c_dict[cancer_type]['A']
	B = c_dict[cancer_type]['B']
	col_inds = [X.columns.get_loc(str(i)) for i in imp.index[A:B].map(str)]
	
	mean_0	= [np.mean(X.values[Ys[cancer_type]==0, col_ind],axis=0) for col_ind in col_inds]
	sd_0	= [np.std(X.values[Ys[cancer_type]==0, col_ind],axis=0) for col_ind in col_inds]
	
	mean_1	= [np.mean(X.values[Ys[cancer_type]==1, col_ind],axis=0) for col_ind in col_inds]
	sd_1	= [np.std(X.values[Ys[cancer_type]==1, col_ind],axis=0) for col_ind in col_inds]
	
	f1 = plt.figure(figsize = (4,12), dpi=600)
	f1.suptitle(cancer_type + ' diff_expression ranks ' + str(A) + ' - ' + str(B), fontsize=12)
	
	ax1 = plt.subplot(511)
	ax1.hist(mean_0, color = 'k', alpha = .3, bins = np.arange(-2,2, .1))
	ax1.hist(mean_1, color = 'r', alpha = .5, bins = np.arange(-2,2, .1))
	plt.xlabel('Mean expression', fontsize=8)
	plt.ylabel('# genes', fontsize=8)
	plt.title('gene set in ' + cancer_type + ' vs rest of data', fontsize=10)
	
	ax2 = plt.subplot(512)
	ax2.hist(sd_0, color = 'k', alpha = .3, bins = np.arange(0,2, .1))
	ax2.hist(sd_1, color = 'b', alpha = .5, bins = np.arange(0,2, .1))
	plt.xlabel('Std', fontsize=8)
	plt.ylabel('# genes', fontsize=8)
	
	ax3 = plt.subplot(513)
	ax3.plot(mean_1, color = 'r', alpha = .5)
	ax3.plot(sd_1, color = 'b', alpha = .5)
	plt.xlabel('rank', fontsize=8)
	plt.ylabel('metric', fontsize=8)
	plt.title('gene set in ' + cancer_type + 'mean (red) and std(blue)', fontsize=10)
	
	# get number of all over/underexpressed genes for particular cancer
	total_oe = np.sum(np.mean(X.values[Ys[cancer_type]==1,:],axis=0) >= oe_thr)
	total_ue = np.sum(np.mean(X.values[Ys[cancer_type]==1,:],axis=0) <= ue_thr)
	
	f_range = range(window_step,c_dict[cancer_type]['B']-c_dict[cancer_type]['A']+window_step,window_step)
		
	oes = np.array([np.sum(np.array(mean_1[:i]) >= oe_thr) for i in f_range])
	ues = np.array([np.sum(np.array(mean_1[:i]) <= ue_thr) for i in f_range])
	
	oe_ratios = oes/total_oe
	ue_ratios = ues/total_ue
	
	ax4 = plt.subplot(514)
	ax4.plot(f_range, oe_ratios, color = 'r')
	ax4.plot(f_range, ue_ratios, color = 'b')
	plt.xlabel('rank', fontsize=8)
	plt.ylabel('fraction', fontsize=8)
	plt.title('Over(red) or under(blue) expressed in\n' + cancer_type + ' as fraction of all genes', fontsize=10)
	
	ax5 = plt.subplot(515)
	ax5.plot(f_range, oes/len(col_inds), color = 'r')
	ax5.plot(f_range, ues/len(col_inds), color = 'b')
	plt.xlabel('rank', fontsize=8)
	plt.ylabel('fraction', fontsize=8)
	plt.title('Over(red) or under(blue) expressed in\n' + cancer_type + ' as fraction of genes in set', fontsize=10)
	
	plt.subplots_adjust(bottom=0.1,
						top=0.9,
						left = 0.2,
						right = 0.8,
						wspace = 0.5,
						hspace = 0.8 )
	
	return {'fig': f1,
			'data1_mean' : mean_1,
			'data1_sd': sd_1,
			'data0_mean': mean_0,
			'data0_sd' : sd_0}

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

print('Loading ID mapping from ' + id_map_file)
id_map = pd.read_csv(id_map_file, delimiter='\t',names=['name','id'])

X = data.iloc[:,4:]
X[X > 10] = 10

Ys = {}
Y = np.zeros((data.shape[0],),dtype=int)
for cancer_type in c_dict.keys():
	Ys[cancer_type] = Y.copy()
	Ys[cancer_type][data['ONCOTREE_CODE'] == cancer_type] = 1
	print("%s samples:\t%d" % (cancer_type, np.sum(Ys[cancer_type],axis=0)) )

for cancer_type in c_dict.keys():
	imp_file_name = cancer_type + fn_middle +'.csv'
	outfile_prefix = cancer_type + '_' + str(c_dict[cancer_type]['A']) + '_' + str(c_dict[cancer_type]['B']) + '_diff_expression'
	
	imp = pd.read_csv(imp_file_name, index_col=0, header=None)
	D = do_plotting(cancer_type = cancer_type, X = X, Ys = Ys, c_dict = c_dict, imp = imp)
	
	f1 = D['fig']
	f1.savefig(outfile_prefix + '.png')
	plt.close(f1)
	
	index2gene = [id_map.loc[id_map['id'] == fn,'name'].values[0] for fn in imp.index[c_dict[cancer_type]['A']:c_dict[cancer_type]['B']].map(str)]
	expr_data = pd.DataFrame(index=index2gene)
	expr_data['mean_0'] = data=D['data0_mean']
	expr_data['sd_0'] = data=D['data0_sd']
	expr_data['mean_1'] = data=D['data1_mean']
	expr_data['sd_1'] = data=D['data1_sd']
	expr_data.to_csv(outfile_prefix + '.csv')
