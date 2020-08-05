########################
#                      #
#      PGM PROJECT     #
#    Ehsan Montahaie   #
#                      #
########################

import random
import numpy as np
import sys
import dataset
from scipy import special
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool

import math

seed=int(file('seed.txt').read())
np.random.seed(seed)
random.seed(seed)
#####################################################################################
def mycaller_func1(x): #help to parallel objects
	# import copy
	# y=copy.deepcopy(x[0])
	# return y.solve(x[1])
	return x[0].solve(x[1])
def mycaller_func2(x): #help to parallel objects
	# import copy
	# y=copy.deepcopy(x[0])
	# return y.solve(x[1],x[2])
	return x[0].solve(x[1],x[2])

#################################
#	___  _  _     ____ _  _ 	#
#	|__] |\/|     |___ |\/| 	#
#	|    |  | ___ |___ |  |		#
#	                       		#
#################################

class PM_EM:
	def __init__(self,observe_a,w,N,K,max_iteration,prefix="",thread_number=2):
		self.observe_a=observe_a
		self.w=w
		self.N=N
		self.K=K
		self.tune=4.0
		self.max_iteration=max_iteration
		self.prefix=prefix
		self.c_min=10**-20
		self.c_max=10**-4 * 2
		self.c_init=np.identity(N)* 10**-4
		self.max_penality=10
		self.thread_number=thread_number

	def _calc_MAE_vector(self,estimated_a):
		defrence=abs(self.observe_a-estimated_a).astype(float)
		zero_mask_orginal_a 	= (self.observe_a==0).astype(float)
		zero_mask_defrence	= (defrence<=0.01).astype(float) #ignore small defrence
		
		invers_input =1.0/(self.observe_a+zero_mask_orginal_a)

		mae_matrix=np.multiply(defrence+zero_mask_defrence,invers_input)
		mae_matrix=np.multiply(mae_matrix,1.0-zero_mask_defrence)
		mae_matrix=np.multiply(mae_matrix,self.w)
		return np.sum(mae_matrix,axis=0)

	def _c_scale_list(self,new_f,last_mae_vector):
		new_estimate_a=new_f*new_f.T
		new_mae_vector=self._calc_MAE_vector(new_estimate_a)

		defrence=new_mae_vector-last_mae_vector
		zero_mask_defrence=(defrence<=0).astype(float)

		scale_error=zero_mask_defrence*self.tune+(1.0-zero_mask_defrence)*(0.5/self.tune)

		return scale_error,new_estimate_a,new_mae_vector

	def _update_formula(self,f,estimated_a,last_c):
		tmp=np.multiply(self.w,1.0/(estimated_a))
		q=np.multiply(self.observe_a,tmp)-self.w
		return (np.identity(self.N)+last_c*q)*f
		#return (np.identity(N)+c*q  -2*c*np.matrix(np.ones( (N,N) )) )*f
		#return (np.identity(N)+c*q-c*np.matrix(np.ones( (N,N) )))*f

	def solve(self,f):

		c=self.c_init.copy()		
		best_f=f.copy()
		estimated_a=f*f.T
		mae_vector=self._calc_MAE_vector(estimated_a)
		best_f_mae=mae_error=mae_vector.sum()/float((self.w!=0).sum())
		good_direction=self.N
		penalty=0
		d=0.0
		it=0
		negative_num=0
		while penalty<self.max_penality and it<self.max_iteration and not np.isnan(f).any() and not np.isinf(f).any():# and negative_num==0:
			it+=1
			if best_f_mae>mae_error:
				best_f,best_f_mae=f.copy(),mae_error
			
			f_old=f.copy()
			f=self._update_formula(f,estimated_a,c)	
			d=np.linalg.norm(f-f_old)

			scale_error,estimated_a,mae_vector=self._c_scale_list(
				new_f=f
				,last_mae_vector=mae_vector
			)

			tune_matrix=np.diag(scale_error.A1)
			c=np.multiply(c,tune_matrix)

			min_mask=(c<self.c_min).astype(float)
			c=min_mask*self.c_min+np.multiply(1.0-min_mask,c)
			max_mask=(c>self.c_max).astype(float)
			c=max_mask*self.c_max+np.multiply(1.0-max_mask,c)
			
			negative_num=(estimated_a<-0.5).sum()
			last_mae_error=mae_error
			mae_error=mae_vector.sum()/float((self.w!=0).sum())
			good_direction=(tune_matrix>1).sum()

			if good_direction==0 or d<10**-10 or mae_error>10**3:
				penalty+=1
			else:
				penalty=0

			if self.prefix!="" and it%50==0:
				print "%s%2d\tMAE(withzeros)=%.10f\tdef=%.10f\tneg=%d\t2X=%d"%(self.prefix,it,mae_error,d,negative_num,good_direction)
				sys.stdout.flush()

		return best_f,best_f_mae
	def solve_parallel(self,f_list):
		conf_list=zip([self]*len(f_list),f_list)
		
		# pool=Pool(self.thread_number)
		# res=pool.map(mycaller_func1,conf_list)

		tp = ThreadPool(processes=self.thread_number)
		res=tp.map(self.solve,f_list)
		tp.close()
		return res

class PM_EM_search_initial:
	def __init__(self,DATASET,w,observe_a,K,population,exchange_max_iteration=20,exchange_iteration=100):
		self.dataset=DATASET
		self.population=population
		self.exchange_max_iteration=exchange_max_iteration
		self.exchange_iteration=exchange_iteration
		self.K=K
		self.N=DATASET.N
		self.EM=PM_EM(observe_a,w,DATASET.N,K,10000,"")
		self.EM.thread_number=population
	def _make_initial_f(self,num):
		f_list=[]
		f_config=[]
		for i in range(num):
			if i%3==0:
				a=np.random.rand()*0.5
				f_list.append(np.matrix(np.ones( (self.N,self.K) ))*a)
				f_config.append(("uniform",a))
			elif i%3==1:
				a=np.random.rand()*10.0
				b=np.random.rand()*10.0
				f_list.append(np.matrix(np.random.gamma(a,b, (self.N,self.K))))
				f_config.append(("gamma",a,b))
			elif i%3==2:
				a=np.random.rand()*10.0
				b=np.random.rand()*10.0
				f_list.append(np.matrix(np.random.normal(a,b, (self.N,self.K))))
				f_config.append(("normal",a,b))
		return f_list,f_config

	def _search_initial(self):
		population=self.population
		
		init_f,init_conf=self._make_initial_f(population)
		state=zip([999999.99999]*population,init_f,init_conf)

		orginal_it_num=self.EM.max_iteration
		self.EM.max_iteration=self.exchange_max_iteration
		for eit in range(self.exchange_iteration+1):
			second_pop=min(population-int(math.floor((eit*population)/self.exchange_iteration)),population-1) # random part
			first_pop=population-second_pop

			#print "exchange %02d: best MAE(withzeros)	: %.5f"%(eit,state[0][0])
			#sys.stdout.flush()
			new_init=self._make_initial_f(second_pop)
			half_new_state=zip([999999.99999]*second_pop,new_init[0],new_init[1])
			state=state[:first_pop]+half_new_state

			last_state_seprated=zip(*state)
			if eit==self.exchange_iteration:
				self.EM.max_iteration=orginal_it_num
				#self.EM.prefix="\t"
			results = self.EM.solve_parallel(last_state_seprated[1])
			f_list,mae_list=zip(*results)
			state=zip(mae_list,f_list,last_state_seprated[2])
			state=sorted(state)

		return state

	def search_best_f(self):
		states=self._search_initial()
		best=list(states[0])
		best[0]=[int(states[0][0]*100),1.0]

		for s in states:
			f=np.matrix(s[1])
			estimated_a=f*f.T
			error=[int(self.dataset.MAE(estimated_a,1)*100),1-self.dataset.accuracy(estimated_a,1)]
			if error<best[0]:
				best=list(s)
				best[0]=error
		print "best initial:",best[2]
		return best[1]

#################################
#	___  _  _     _  _ ____ 	#
#	|__] |\/|     |  | |___ 	#
#	|    |  | ___  \/  |___ 	#
#	                        	#
#################################


class PM_VE:
	def __init__(self,observe_a,w,N,K,thread_number=2):
		self.observe_a=observe_a
		self.w=w
		self.N=N
		self.K=K
		self.thread_number=thread_number

	def solve(self,a,b):
		#print a,b
		N=self.N
		K=self.K
		ru=np.random.random( (N,N,K) ).astype(float)
		lamb=np.random.random( (N,N,K) ).astype(float)
		alpha=np.random.random( (N,K) ).astype(float)
		beta=np.random.random( (N,K) ).astype(float)

		for m in range(N):
			for n in range(m):
				lamb[m,n,:]=lamb[n,m,:]
		d=1.0
		it=0
		while d>=10**-3 and it<50 and not np.isnan(lamb).any() and not np.isinf(lamb).any():
			it+=1
			# old_ru 	 	=ru.copy()
			old_lamb 	=lamb.copy()
			# old_alpha 	=alpha.copy()
			# old_beta 	=beta.copy()

			#precompute E[f]
			E_f=special.digamma(alpha)-np.log(beta)
			###################################
			#update lambda
			for m in range(N):
				for n in range(m):
					lamb[n,m,:]=lamb[m,n,:]=np.exp(E_f[m,:]+E_f[n,:])
			###################################
			#update ru
			for m in range(N):
				for n in range(m):
					ru[n,m,:]=ru[m,n,:]=lamb[m,n,:]/(lamb[m,n,:].sum())
			#cprecompute E[c]
			E_c=lamb.copy()
			for m in range(N):
				for n in range(m):
					if w[m,n]==1:
						E_c[n,m,:]=E_c[m,n,:]=self.observe_a[m,n]*ru[m,n,:]
			###################################
			#update alpha
			tmp_sum=np.zeros((1,K)).astype(float)
			for m in range(N):
				tmp_sum[0,:]+=E_c[m,n,:]
			for n in range(N):
				alpha[n,:]=a+tmp_sum[0,:]-E_c[n,n,:]
			###################################
			#update beta
			# tmp_sum=np.zeros((1,K)).astype(float)
			# for m in range(N):
			# 	tmp_sum+=alpha[m,:]/beta[m,:]
			for n in range(N):
				beta[n,:]=b
				for m in range(N):
					if m==n:continue
					beta[n,:]+=(alpha[m,:]/beta[m,:])

			d=np.linalg.norm(old_lamb-lamb)			
			#print a,b,">",d;sys.stdout.flush()
		estimated_a=np.matrix(np.zeros( (N,N) )).astype(float)
		for i in range(K):
			tmp=np.matrix(lamb[:,:,i])
			estimated_a[:,:]+=(tmp+tmp.T)/2.0
		return np.matrix(estimated_a)
		#print a,b,DATASET.MAE(estimated_a,1),DATASET.accuracy(estimated_a,1)
		#DATASET.histogram(estimated_a)
	
	def search_best_A(self,dataset,try_num):
		conf_list=[]
		for _ in range(try_num):
			mean=np.random.rand()*20.0+1.0
			var=np.random.rand()*20+2.0
			beta=mean/var
			alpha=mean*beta
			conf_list.append( (self,alpha,beta) )
		
		pool=Pool(self.thread_number)
		res=pool.map(mycaller_func2,conf_list)
		for i in range(try_num):
			conf_list[i]=(conf_list[i][1],conf_list[i][2])
		# tp = ThreadPool(processes=self.thread_number)
		# res=tp.map(self.solve,conf_list)
		# tp.close()

		best_mae 	=[999.999,0]
		best_acc 	=[999.999,0]
		best_ma 	=[999.999,0]
		for i,r in enumerate(res):			
			mae=dataset.MAE(r,1)
			acc=1-dataset.accuracy(r,1)
			ma=(mae+acc)
			if mae<best_mae[0]:
				best_mae=[mae,i]
			if mae<best_acc[0]:
				best_acc=[acc,i]
			if mae<best_ma[0]:
				best_ma=[ma,i]
		return 	(conf_list[best_mae[1]],res[best_mae[1]]), (conf_list[best_acc[1]],res[best_acc[1]]), (conf_list[best_ma[1]],res[best_ma[1]])



#################################
#	___  _  _     ____ _  _ 	#
#	|__] |\/|     [__  |\/| 	#
#	|    |  | ___ ___] |  | 	#
#	                        	#
#################################



class PM_SM:
	def __init__(self,observe_a,w,N,K,thread_number=2):
		self.observe_a=observe_a
		self.w=w
		self.N=N
		self.K=K
		self.thread_number=thread_number
		self.burn_in=50
		self.sampling=150

	def solve(self,a,b):
		#print a,b
		N=self.N
		K=self.K
		sample_f 	=np.random.random( (N,K) ).astype(float)
		sample_c 	=np.random.random( (N,N,K) ).astype(float)

		sum_c=np.zeros( (N,N,K) ).astype(float)

		cnt=0
		for it in range(self.sampling+self.burn_in):
			for l in range(K):
				for n in range(N):
					alpha 	=a+	sample_c[:,n,l].sum()	-sample_c[n,n,l]
					beta 	=b+	sample_f[:,l].sum() 	-sample_f[n,l]
					sample_f[n,l]=np.random.gamma(alpha,1.0/float(beta))

			for m in range(N):
				for n in range(m):
					if w[m,n]==0:
						for l in range(K):
							lamb=sample_f[m,l]*sample_f[n,l]
							sample_c[n,m,l]=sample_c[m,n,l]=np.random.poisson(lamb)
					else:
						ps=np.multiply(sample_f[m,:],sample_f[n,:])
						ps=ps/ps.sum()
						ps=list(ps)
						sample_c[n,m,:]=sample_c[m,n,:]=np.random.multinomial(self.observe_a[m,n],ps)
			if it>self.burn_in: #after burn in  
				sum_c+=sample_c
				cnt+=1
		avg_c=sum_c/float(cnt)
		estimated_a=np.zeros( (N,N) )
		for m in range(N):
			for n in range(N):
					estimated_a[m,n]=avg_c[m,n,:].sum()
		return estimated_a
	
	def search_best_A(self,dataset,try_num):
		conf_list=[]
		for _ in range(try_num):
			mean=np.random.rand()*20.0+1.0
			var=np.random.rand()*20+2.0
			beta=mean/var
			alpha=mean*beta
			conf_list.append( [self,alpha,beta] )
		# tp = ThreadPool(processes=self.thread_number)
		# res=tp.map(self.solve,conf_list)
		# tp.close()
		pool=Pool(self.thread_number)
		res=pool.map(mycaller_func2,conf_list)
		for i in range(try_num):
			conf_list[i]=(conf_list[i][1],conf_list[i][2])

		best_mae 	=[999.999,0]
		best_acc 	=[999.999,0]
		best_ma 	=[999.999,0]
		for i,r in enumerate(res):
			mae=dataset.MAE(r,1)
			acc=1-dataset.accuracy(r,1)
			ma=(mae+acc)
			if mae<best_mae[0]:
				best_mae=[mae,i]
			if mae<best_acc[0]:
				best_acc=[acc,i]
			if mae<best_ma[0]:
				best_ma=[ma,i]
		return 	(conf_list[best_mae[1]],res[best_mae[1]]), (conf_list[best_acc[1]],res[best_acc[1]]), (conf_list[best_ma[1]],res[best_ma[1]])


###############################################################################################################

###############################################################################################################


def print_error(dataset,estimated_a):
	for i in [1]:
		print "\t%s : mae=%.3f accuracy=%2.1f%%" % (["train:","test:"][i],dataset.MAE(estimated_a,i),dataset.accuracy(estimated_a,i)*100)
		sys.stdout.flush()

THREAD_NUMBER=60
DATASET=dataset.dataset('./DataSet2.txt') # small dataset
#DATASET=dataset.dataset('./DATASET.txt') # large dataset
N=DATASET.N

KLIST=[4,8,10,12,16,20]
WLIST=[0.1,0.2,0.3,0.4,0.5]
for K,miss_data in zip([12]*len(WLIST),WLIST)+zip(KLIST,[0.2]*len(KLIST)):
	print "-"*50
	print "K=",K,"miss data=",miss_data
	w,observe_a=DATASET.generate(miss_data)
	phase1=PM_EM_search_initial(DATASET,w,observe_a,K,THREAD_NUMBER)
	f=phase1.search_best_f()
	estimated_a=f*f.T
	print_error(DATASET,estimated_a)

	phase21=PM_VE(observe_a,w,N,K,THREAD_NUMBER)
	config,estimated_a=phase21.search_best_A(DATASET,1000)[0]
	print "a,b=",config
	print_error(DATASET,estimated_a)

	phase22=PM_SM(observe_a,w,N,K,THREAD_NUMBER)
	config,estimated_a=phase22.search_best_A(DATASET,1000)[0]
	print "a,b=",config
	print_error(DATASET,estimated_a)
