import numpy as np

class PSO():
	def __init__(self,pN,dim,max_iter,func):  
		self.w = 0.8    			#惯性因子
		self.c1 = 2     			#自身认知因子
		self.c2 = 2     			#社会认知因子
		self.r1 = 0.6  				#自身认知学习率
		self.r2 = 0.3  				#社会认知学习率
		self.pN = pN                #粒子数量
		self.dim = dim              	#搜索维度  
		self.max_iter = max_iter    				#最大迭代次数  
		self.X = np.zeros((self.pN,self.dim))       #初始粒子的位置和速度 
		self.V = np.zeros((self.pN,self.dim)) 
		self.pbest = np.zeros((self.pN,self.dim),dtype = float)   #粒子历史最佳位置
		self.gbest = np.zeros((1,self.dim),dtype = float)  		#全局最佳位置  
		self.p_bestfit = np.zeros(self.pN)              #每个个体的历史最佳适应值  
		self.fit = -1e15             				#全局最佳适应值  
		self.func = func
	
	def function(self,x):
		return self.func(x)

	def init_pop(self,):  #初始化种群  
		for i in range(self.pN):  
			#初始化每一个粒子的位置和速度
			self.X[i] = np.random.uniform(0,5,[1,self.dim])  
			self.V[i] = np.random.uniform(0,5,[1,self.dim])  

			self.pbest[i] = self.X[i]  #初始化历史最佳位置
			self.p_bestfit[i] = self.function(self.X[i])  #得到对应的fit值
		if(self.p_bestfit[i] > self.fit):  
			self.fit = self.p_bestfit[i] 
			self.gbest = self.X[i]  	#得到全局最佳

	def update(self):  
		fitness = []  

		for _ in range(self.max_iter):  
			for i in range(self.pN):         #更新gbest\pbest  
				temp = self.function(self.X[i])  #获得当前位置的适应值
				if( temp > self.p_bestfit[i] ):      #更新个体最优  
					self.p_bestfit[i] = temp  
					self.pbest[i] = self.X[i]  
					if(self.p_bestfit[i] > self.fit):  #更新全局最优  
						self.gbest = self.X[i]  
						self.fit = self.p_bestfit[i]  

			for i in range(self.pN):  	#更新权重
				self.V[i] = self.w*self.V[i] + self.c1*self.r1*(self.pbest[i] - self.X[i]) + \
							self.c2*self.r2*(self.gbest - self.X[i])  
				self.X[i] = self.X[i] + self.V[i]  

			fitness.append(self.fit)  

		return self.gbest,self.fit

def count_func(x):
	y = -x**2 + 20*x + 10
	return y 

pso_example = PSO(pN = 50,dim = 1,max_iter = 300, func = count_func)
pso_example.init_pop()
x_best,fit_best= pso_example.update()
print(x_best,fit_best)

