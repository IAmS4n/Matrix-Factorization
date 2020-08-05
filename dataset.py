import random
import numpy as np
import sys
seed=int(file('seed.txt').read())
np.random.seed(seed)
random.seed(seed)
#####################################################################################

class dataset:
	def __init__(self,path):
		self.orginal_a=[]
		for line in file(path).read().split("\n")[:-1]:
			self.orginal_a.append(map(int,line.split(",")))
		self.orginal_a=np.matrix(self.orginal_a)
		self.orginal_a=self.orginal_a.astype(float)
		self.N = len(self.orginal_a)

		self.i_orginal_a=(self.orginal_a>0.5).astype(float)

	def _random_inc_pair(self,bound):
		while True:
			x,y=sorted([random.randint(0,bound-1),random.randint(0,bound-1)])
			if x!=y:
				return x,y

	def _check_symetric(self,inp,tol=1e-2):
		return np.allclose(inp,inp.T, atol=tol)

	def generate(self,miss): # w :symteric , diagonal zero 
		N=self.N
		w=np.zeros((N,N))
		p_one=1-miss
		for _ in range(int(0.5*p_one*N*N)):
			while True:
				x,y=self._random_inc_pair(N)
				if w[x][y]!=1:
					break
			w[y][x]=w[x][y]=1
		self.w=np.matrix(w).astype(float)
		if not self._check_symetric(self.w) :
			print "[-]INVALID RANDOM W"

		self.observe_a=np.multiply(self.w,self.orginal_a).astype(float)
		
		upper_trangle=np.triu(np.ones((N,N)), 1)
		self.filter_observe_a 	=np.multiply(self.w,upper_trangle)
		self.filter_latent_a 	=np.multiply(1.0-self.w,upper_trangle)

		return self.w,self.observe_a
	

	def accuracy(self,estimated_a,test=0):
		# if not self._check_symetric(estimated_a) :
		# 	print "[-]INVALID ESTIMATED A"

		if test==0: # train
			data_filter=self.filter_observe_a
		else:
			data_filter=self.filter_latent_a

		i_estimated_a=(estimated_a>0.5).astype(float)

		filtered_i_estimated_a 	=np.multiply(data_filter, i_estimated_a)
		filtered_i_source_a 	=np.multiply(data_filter, self.i_orginal_a)

		i_defrence=abs(filtered_i_source_a-filtered_i_estimated_a)
		i_sum=filtered_i_source_a+filtered_i_estimated_a

		return 1.0-float(i_defrence.sum())/float(i_sum.sum())

	def MAE(self,estimated_a,test=0):
		# if not self._check_symetric(estimated_a) :
		# 	print "[-]INVALID ESTIMATED A"

		if test==0: # train
			data_filter=self.filter_observe_a
		else:
			data_filter=self.filter_latent_a

		filtered_source_a =np.multiply(data_filter, self.orginal_a)
		filtered_absdefrence=abs(estimated_a-filtered_source_a)

		mae=0.0
		cnt=0
		for i in range(self.N):
			for j in range(self.N):
				if data_filter[i,j]<=0.00001:continue
				if self.orginal_a[i,j]<=0.5:continue
				
				mae+=filtered_absdefrence[i,j]/filtered_source_a[i,j]
				cnt+=1

		return mae/float(cnt)
	def histogram(self,estimated_a):
		import matplotlib.mlab as mlab
		import matplotlib.pyplot as plt
		plt.hist(list(self.orginal_a.A1), 50, normed=1, alpha=0.75)
		plt.hist(list(np.matrix(estimated_a).A1), 50, normed=1, alpha=0.75)
		plt.grid(True)
		plt.show()