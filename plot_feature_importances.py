import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ml_utils

fn_middle		= sys.argv[1]
cancer_types	= sys.argv[2:]

for cancer_type in cancer_types:
	f_prefix = cancer_type + fn_middle
	fi_df = pd.read_csv(f_prefix +'.csv', index_col=0, header=None)
	
	f1 = ml_utils.plot_feature_importances(name = cancer_type, imps=fi_df)
	f1.savefig(f_prefix + '.png')
	plt.close(f1)
