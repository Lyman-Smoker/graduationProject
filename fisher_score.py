import numpy as np

def fisher_z(score_list):
	score_list = np.array(score_list)
	z_transform = 0.5 * np.log((1+score_list)/(1-score_list))
	mean_z = np.mean(z_transform)
	final_score = (np.e**(2*mean_z)-1) / (np.e**(2*mean_z)+1)
	return final_score

li = [0.86, 0.8050,0.6483,0.8757,0.9330,0.9316]
print(fisher_z(li))