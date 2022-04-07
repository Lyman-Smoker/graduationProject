import numpy as np

def fisher_z(score_list):
	score_list = np.array(score_list)
	z_transform = 0.5 * np.log((1+score_list)/(1-score_list))
	mean_z = np.mean(z_transform)
	final_score = (np.e**(2*mean_z)-1) / (np.e**(2*mean_z)+1)
	return final_score

li = [0.8802,0.8041,0.6612,0.5707,0.9126,0.9088]
print(fisher_z(li))