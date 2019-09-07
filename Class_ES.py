import numpy as np
import matplotlib.pyplot as plt

# 每个个体的长度
GENE_SIZE = 1
# 每个基因的范围
GENE_BOUND = [0, 5]    
# 200代   
N_GENERATIONS = 200
# 种群的大小
POP_SIZE = 100          
# 每一代生成50个孩子
N_KID = 50  

# 寻找函数的最大值
def F(x): 
    return np.sin(10*x)*x + np.cos(2*x)*x    

class ES():
    def __init__(self,gene_size,pop_size,n_kid):
        # 基因长度代表字符串的长度
        self.gene_size = gene_size
        # 种群的大小代表种群中有几个个体
        self.pop_size = pop_size
        self.n_kid = n_kid
        self.init_pop()
        print(self.pop)
    # 降到一维
    def get_fitness(self): 
        return self.pred.flatten()

    # 初始化种群
    def init_pop(self):
        self.pop = dict(DNA=5 * np.random.rand(1, self.gene_size).repeat(POP_SIZE, axis=0),
           mut_strength=np.random.rand(POP_SIZE, self.gene_size))

    # 更新后代
    def make_kid(self):
        # DNA指的是当前孩子的基因
        # mut_strength指的是变异强度
        self.kids = {'DNA': np.empty((self.n_kid, self.gene_size)),
                'mut_strength': np.empty((self.n_kid, self.gene_size))}

        for kv, ks in zip(self.kids['DNA'], self.kids['mut_strength']):
            # 杂交，随机选择父母
            p1, p2 = np.random.choice(self.pop_size, size=2, replace=False)
            # 选择杂交点
            cp = np.random.randint(0, 2, self.gene_size, dtype=np.bool)
            # 当前孩子基因的杂交结果
            kv[cp] = self.pop['DNA'][p1, cp]
            kv[~cp] = self.pop['DNA'][p2, ~cp]
            # 当前孩子变异强度的杂交结果
            ks[cp] = self.pop['mut_strength'][p1, cp]
            ks[~cp] = self.pop['mut_strength'][p2, ~cp]

            # 变异强度要大于0，并且不断缩小
            ks[:] = np.maximum(ks + (np.random.rand()-0.5), 0.)    
            kv += ks * np.random.randn()
            # 截断
            kv[:] = np.clip(kv,GENE_BOUND[0],GENE_BOUND[1])   

    # 淘汰低适应度后代
    def kill_bad(self):
        # 进行vertical垂直叠加
        for key in ['DNA', 'mut_strength']:
            self.pop[key] = np.vstack((self.pop[key], self.kids[key]))

        # 计算fitness
        self.pred = F(self.pop['DNA'])
        fitness = self.get_fitness()
        
        # 读出按照降序排列fitness的索引
        max_index = np.argsort(-fitness)
        # 选择适应度最大的50个个体
        good_idx = max_index[:POP_SIZE]   
        for key in ['DNA', 'mut_strength']:
            self.pop[key] = self.pop[key][good_idx]


test1 = ES(gene_size = GENE_SIZE,pop_size = POP_SIZE,n_kid = N_KID)

plt.ion()     
x = np.linspace(*GENE_BOUND, 200)
plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
    # 画图部分
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(test1.pop['DNA'], F(test1.pop['DNA']), s=200, lw=0, c='red', alpha=0.5)
    plt.pause(0.05)

    # ES更新
    kids = test1.make_kid()
    pop = test1.kill_bad()

plt.ioff(); plt.show()