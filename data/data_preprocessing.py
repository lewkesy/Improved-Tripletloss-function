import random
import os

# Do not run these functions!!!

def main():

			
def move_pics(self, parameter_list):
	num_samples_cate=np.array(constrained_sum_sample_pos(102, 1829))
	for i in range(102):
		num_each=np.random.randint(len(visited[d_dict[i]]),size=18)
		for idx in num_each:
			shutil.copy('../data/train/'+ d_dict[i] + '/' + visited[d_dict[i]][idx],'../data/test/'+visited[d_dict[i]][idx])


def constrained_sum_sample_pos(n, total):
    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]
    return v_dict, d_dict

def rename_pics():
	for dirname in os.listdir('../data/train'):
			index=0
			for filename in os.listdir('../data/train/'+dirname):
				os.rename('../data/train/'+str(dirname)+'/'+str(filename), '../data/train/'+str(dirname)+'/'+str(dirname)+'_'+str(index)+'.jpg')
				index+=1
			

if __name__ == "__main__": main()