
#-----------------------GERT网络的矩阵式求解算法---------------------------

#该算法的性能瓶颈主要在矩阵代数余子式求解过程。如果网络过于复杂，存在多个代数余子式的求解需求(始节点的作用节点有多个)，可采用多进程提高运算速度。

#该算法理论部分参考：[1]陶良彦,刘思峰,方志耕,等.GERT网络的矩阵式表达及求解模型[J].系统工程与电子技术,2017,39(06):1292-1297.

#该算法的传递函数默认为正态分布

#------------------------------------------------------------------------


import networkx as nx
from functools import lru_cache
import pandas as pd
import numpy as np
import sympy


#该类有5个方法：
# Formula()获取符号函数公式，
# FormulaExp()获取实际函数公式，
# Probability()获取概率值，
# Value()获取传递值,
# latex()导出Latex格式公式


class GertMatrixCalcul:
    def __init__(self,filepath,source_code,sink_code):
        self.filepath=filepath  #GERT数据存放路径
        self.source_code=source_code  #起始节点代号
        self.sink_code=sink_code  #终止节点代号
    
    @lru_cache(maxsize=None) #使用装饰器实现缓存
    def Matrix(self):
        df=pd.read_excel(self.filepath) #读取GERT数据文件
        start=df['start'] 
        end=df['end']
        norm_ave=df['ave'] #正态分布均值
        norm_var=df['var'] #生态分布方差
        norm_p=df['p']     #传递概率
        fun_dict={}
        s=sympy.symbols('s')
        edges=[]
        for i in range(len(start)):
            sym_smb=sympy.symbols(f"w{start[i]}{end[i]}")
            fun=norm_p[i]*sympy.exp(norm_ave[i]*s+(norm_var[i]/2) * (s ** 2))
            value=fun.subs(s,0)  
            fun_dict[sym_smb]=(fun,value)
            edges.append((start[i],end[i])) 
        G=nx.MultiDiGraph()
        G.add_edges_from(edges)

        nodelist=sorted(G.nodes)
        np_matrix=nx.to_numpy_array(G,nodelist=nodelist,dtype=object).T
        rows, cols = np.nonzero(np_matrix)
        for i in range(len(rows)):
            row, col = nodelist[rows[i]], nodelist[cols[i]]
            sym=sympy.Symbol(f"w{col}{row}")
            np_matrix[rows[i],cols[i]]=sym


        source_nodes = [node for node, deg in G.in_degree() if deg == 0]
        sink_nodes = [node for node, deg in G.out_degree() if deg == 0]

        sink_nodes.remove(self.sink_code)
        deleted=source_nodes+sink_nodes
        index_deleted=[nodelist.index(i) for i in deleted]
        source_index=nodelist.index(self.source_code)

        source_next=np_matrix[:,source_index][np.nonzero(np_matrix[:,source_index])]
        b=-np.array(source_next)
        
        origin_next_index=np.nonzero(np_matrix[:,source_index])[0].tolist()
        
        origin_tagrget_index=nodelist.index(self.sink_code)

        final_next_index=origin_next_index.copy()

        for index in sorted(index_deleted):
            if index<origin_tagrget_index:
                origin_tagrget_index-=1
            for i,next_index in enumerate(origin_next_index):
                if index<next_index:
                    final_next_index[i]-=1

        temp_matrix=np.delete(np_matrix,index_deleted,axis=0)
        new_matrix=np.delete(temp_matrix,index_deleted,axis=1)

        position=[(i,origin_tagrget_index) for i in final_next_index]

        return new_matrix,b,fun_dict,position
    
    @lru_cache(maxsize=None) #使用装饰器实现缓存
    def Formula(self): #获取整个网络的传递函数公式
        np_matrix,b,fun_dict,position=self.Matrix()
        new_matrix=np_matrix.copy()  #复制一份numpy矩阵
        np.fill_diagonal(new_matrix,new_matrix.diagonal()-1)  #矩阵主对角线上元素减1
        sym_matrix=sympy.Matrix(new_matrix) 
        ns_matrix=sympy.nsimplify(sym_matrix)
        b=sympy.Matrix(b)
        A_det=ns_matrix.det(method='bareiss') #矩阵行列式

        cofactors=[]
        for item in position:
            cofactors.append(ns_matrix.cofactor(item[0],item[1],method='bareiss')) #矩阵代数余子式求解

        sym_cofactors=sympy.Matrix(cofactors)
        formula=sym_cofactors.dot(b)/A_det

        simplified_formula=sympy.simplify(formula) #简化函数表达式
        return simplified_formula
    
    def Probability(self):

        np_matrix,b_matrix,fun_dict,position=self.Matrix()
        new_matrix=np_matrix.copy()
        new_b=b_matrix.copy()
        for index in np.ndindex(new_matrix.shape):
            element=new_matrix[index]
            if isinstance(element,sympy.Basic):
                new_matrix[index]=fun_dict[element][1] #将符号变量替换为实际的值
        
        for index in np.ndindex(new_b.shape):
            element=-new_b[index]
            new_b[index]=-fun_dict[element][1]

        value_matrix=new_matrix.astype(np.float64) 
        np.fill_diagonal(value_matrix,value_matrix.diagonal()-1)
        value_b=new_b.astype(np.float64)

        A_det=np.linalg.det(value_matrix)
        cofactors=[]
        for item in position:
            minor_matrix = np.delete(np.delete(value_matrix, item[0], axis=0), item[1], axis=1)
            minor_det = np.linalg.det(minor_matrix)
            sign=(-1)**(item[0]+item[1])
            cofactor=sign*minor_det
            cofactors.append(cofactor)
        return np.dot(cofactors,value_b)/A_det
    
    def Value(self): #传递值

        formula=self.Formula()
        fun_dict=self.Matrix()[2]

        sub_fun={}
        for key,value in fun_dict.items():
            sub_fun[key]=value[0]
        replaced_formula=formula.subs(sub_fun)
        s=sympy.Symbol('s')
        f1=sympy.diff(replaced_formula,s) #求s的一阶偏导
        result=f1.subs({s:0}).evalf() #s等于0时，传递函数的值
        return result
    
    def FormulaExp(self): #具体的传递函数公式

        formula=self.Formula()
        fun_dict=self.Matrix()[2]
        sub_fun={}
        for key,value in fun_dict.items():
            sub_fun[key]=value[0]
        s=sympy.Symbol('s')
        replaced_formula=formula.subs(sub_fun)
        return replaced_formula
    
    def latex(self,filename): #导出Latex公式文件，文件名需自命名
        formula1=self.Formula()
        formula2=self.FormulaExp()
        latex_expr1=sympy.latex(formula1)
        latex_expr2=sympy.latex(formula2)
        with open(filename+'.tex', 'w') as f:
            f.write('\\documentclass{article}\n')
            f.write('\\begin{document}\n')
            f.write(latex_expr1 + '\n')
            f.write(latex_expr2 + '\n')
            f.write('\\end{document}')


filepath='test_g.xlsx' #GERT数据存放路径
gert=GertMatrixCalcul(filepath,source_code=1,sink_code=3) #实例化，输入三个参数：GERT数据路径，source_code始节点代号，sink_code终节点代号


print('传递函数代数式:',gert.Formula())
print('传递函数公式:',gert.FormulaExp())
print('传递概率',gert.Probability())
print('传递值：',gert.Value()) 

