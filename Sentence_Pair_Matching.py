import numpy as np
import math
import random

TARGET_PHRASE = 'XJQ2WX'            # 目标字符串
POP_SIZE = 300                      # 种群数量
CROSS_RATE = 0.4                    # 杂交几率
MUTATION_RATE = 0.01                # 突变几率
N_GENERATIONS = 1000                # 繁殖代数

GENE_SIZE = len(TARGET_PHRASE)      # 获取字符串长度
TARGET_ASCII = np.fromstring(TARGET_PHRASE, dtype=np.uint8)  # 从字符串转换到数字

class GA():
    def __init__(self,gene_size,unit_length,cross_rate,mutation_rate,pop_size):
        # 基因长度代表字符串的长度
        self.gene_size = gene_size
        # 种群的大小代表种群中有几个个体
        self.pop_size = pop_size
        # 单位长度代表字符串每个字符所对应的编码长度
        self.unit_length = unit_length
        # 杂交几率
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.pop = self.init_pop()

    # 初始化种群
    def init_pop(self):
        pop = []
        for _ in range(self.pop_size):
            pop.append(np.random.randint(0,2,size = self.gene_size*self.unit_length)) 
        return np.array(pop)
    
    # 将种群从[1,0,1....,1,0]编码转化成对应的数字
    def pop_translate(self):
        pop_translate = []
        for i in range(self.pop_size):
            unit = []
            # 读出一个个体基因中每一段对应的数字
            for j in range(self.gene_size):
                low,high = j*self.unit_length,(j+1)*self.unit_length
                after = self.binarytodecimal(self.pop[i,low:high])
                unit.append(after)
            # 存入到转换后的种群中
            pop_translate.append(unit)
        return np.array(pop_translate,dtype = np.int8) 

    # 该部分用于将字符对应的编码[1,0,1....,1,0]转换成对应的数字
    def binarytodecimal(self,binary):
        total = 0
        for j in range(len(binary)):
            total += binary[j] * (2**j)
        total = np.int8(total * 94 / 255 + 32)
        return total

    # 计算适应度
    def count_fitness(self):
        # 首先获取转换后的种群
        self.pop_t = self.pop_translate()
        # 将转换后的种群与目标字符串对应的ASCII码比较，计算出适应度
        fitness = (self.pop_t == TARGET_ASCII).sum(axis=1)
        return np.array(fitness)

    # 自然选择
    def selection(self):
        self.fit_value = self.count_fitness()
        [bestindividual, bestfit] = self.best()
        # 利用np.random.choice实现轮盘赌
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=self.fit_value/self.fit_value.sum())
        self.pop = self.pop[idx]
        return [bestindividual, bestfit]

    # 选出最大value对应的索引，取出个体与对应的适应度
    def best(self): 
        index = np.argmax(self.fit_value)
        bestfit = self.fit_value[index]
        print(self.pop[index])
        bestindividual = self.pop_t[index].tostring().decode('ascii')
        return [bestindividual, bestfit]

    # 杂交
    def crossover(self):
        # 按照一定概率杂交
        for i in range(self.pop_size):
            # 判断是否达到杂交几率
            if (np.random.rand() < self.cross_rate):
                # 随机选取杂交点，然后交换结点的基因
                individual_size = self.gene_size*self.unit_length
                # 随机选择另一个个体进行杂交
                destination = np.random.randint(0,self.pop_size)
                # 生成每个基因进行交换的结点
                crosspoint = np.random.randint(0,2,size = individual_size).astype(np.bool) 
                # 进行赋值
                self.pop[i,crosspoint] = self.pop[destination,crosspoint]

    # 基因突变
    def mutation(self):
        # 判断每一条染色体是否需要突变
        for i in range(self.pop_size):
            for j in range(self.gene_size):
                if (np.random.rand() < self.mutation_rate):
                    low,high = j*self.unit_length,(j+1)*self.unit_length
                    self.pop[i,low:high] = np.random.randint(0,2,size=self.unit_length)

    def evolve(self):
        [bestindividual, bestfit] =self.selection()
        self.mutation()
        self.crossover()
        return [bestindividual, bestfit]

# 初始化类
test1 = GA(gene_size = GENE_SIZE,unit_length = 8,cross_rate = CROSS_RATE,
    mutation_rate = MUTATION_RATE,pop_size = POP_SIZE)

results = []

for i in range(N_GENERATIONS):
    [bestindividual, bestfit] = test1.evolve()
    results.append([bestfit,bestindividual])
    print("century：",i,"get：",bestindividual)
    if(bestindividual == TARGET_PHRASE):
        break
results.sort()	
print(results[-1]) #打印使得函数取得最大的个体，和其对应的适应度
