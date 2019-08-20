import numpy as np
import math
import random


def binarytodecimal(binary): # 将二进制转化为十进制，x的范围是[0,10]
	total = 0
	for j in range(len(binary)):
		total += binary[j] * (2**j)
	total = total * 10 / 1023
	return total

def pop_b2d(pop):   # 将整个种群转化成十进制
    temppop = []
    for i in range(len(pop)):
        t = binarytodecimal(pop[i])
        temppop.append(t)
    return temppop

def calobjvalue(pop): # 计算目标函数值
    x = np.array(pop_b2d(pop))
    return count_function(x)

def count_function(x):
    y = np.sin(x) + np.cos(5 * x) - x**2 + 2*x
    return y

def calfitvalue(objvalue):
    # 转化为适应值，目标函数值越大越好
    # 在本例子中可以直接使用函数运算结果为fit值，因为我们求的是最大值
    # 在实际应用中需要处理
    for i in range(len(objvalue)):
        if objvalue[i] < 0:
            objvalue[i] = 0
    return objvalue

def best(pop, fitvalue): 
    #找出适应函数值中最大值，和对应的个体
	bestindividual = pop[0]
	bestfit = fitvalue[0]
	for i in range(1,len(pop)):
		if(fitvalue[i] > bestfit):
			bestfit = fitvalue[i]
			bestindividual = pop[i]
	return [bestindividual, bestfit]

def selection(pop, fit_value):
    probability_fit_value = []
    # 适应度总和
    total_fit = sum(fit_value)
    # 求每个被选择的概率
    probability_fit_value = np.array(fit_value) / total_fit

    # 概率求和排序
    cum_sum_table = cum_sum(probability_fit_value)

    # 获取与pop大小相同的一个概率矩阵，其每一个内容均为一个概率
    # 当概率处于不同范围时，选择不同的个体。
    choose_probability = np.sort([np.random.rand() for i in range(len(pop))])
        
    fitin = 0
    newin = 0
    newpop = pop[:]
    # 轮盘赌法
    while newin < len(pop):
        # 当概率处于不同范围时，选择不同的个体。
        # 如个体适应度分别为1，2，3，4，5的种群
        # 利用np.random.rand()生成一个随机数，当其处于0-0.07时			
        # 选择个体1，当其属于0.07-0.2时选择个体2，以此类推。			
        if (choose_probability[newin] < cum_sum_table[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    # pop里存在重复的个体
    pop = newpop[:]

def cum_sum(fit_value):
    # 输入[1, 2, 3, 4, 5]，返回[1,3,6,10,15]
    temp = fit_value[:]
    temp2 = fit_value[:]
    for i in range(len(temp)):
        temp2[i] = (sum(temp[:i + 1]))
    return temp2
    
def crossover(pop, pc):
    # 按照一定概率杂交
    pop_len = len(pop)
    for i in range(pop_len - 1):
        # 判断是否达到杂交几率
        if (np.random.rand() < pc):
            # 随机选取杂交点，然后交换结点的基因
            individual_size = len(pop[0])
            # 随机选择另一个个体进行杂交
            destination = np.random.randint(0,pop_len)
            # 生成每个基因进行交换的结点
            crosspoint = np.random.randint(0,2,size = individual_size)
            # 找到这些结点的索引
            index = np.argwhere(crosspoint==1)
            # 进行赋值
            pop[i,index] = pop[destination,index]

def mutation(pop, pm):
    pop_len = len(pop)
    individual_size = len(pop[0])
    # 每条染色体随便选一个杂交
    for i in range(pop_len):
        for j in range(individual_size):
            if (np.random.rand() < pm):
                if (pop[i][j] == 1):
                    pop[i][j] = 0
                else:
                    pop[i][j] = 1
                    

# 用遗传算法求函数最大值，组合交叉的概率时0.6，突变的概率为0.001
# y = np.sin(x) + np.cos(5 * x) - x**2 + 2*x


popsize = 50    # 种群的大小
pc = 0.6        # 两个个体交叉的概率
pm = 0.001      # 基因突变的概率
gene_size = 10  # 基因长度为10
generation = 100     # 繁殖100代

results = []
bestindividual = []
bestfit = 0
fitvalue = []

pop = np.array([np.random.randint(0,2,size = gene_size)  for i in range(popsize)])

for i in range(generation):     # 繁殖100代
	objvalue = calobjvalue(pop) # 计算种群中目标函数的值
	fitvalue = calfitvalue(objvalue)    # 计算个体的适应值
	[bestindividual, bestfit] = best(pop, fitvalue)     # 选出最好的个体和最好的适应值
	results.append([bestfit,binarytodecimal(bestindividual)]) # 每次繁殖，将最好的结果记录下来
	selection(pop, fitvalue) # 自然选择，淘汰掉一部分适应性低的个体
	crossover(pop, pc)  # 交叉繁殖
	mutation(pop, pc)   # 基因突变
	
results.sort()	
print(results[-1]) #打印使得函数取得最大的个体，和其对应的适应度
