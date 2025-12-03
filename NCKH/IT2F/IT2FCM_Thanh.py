import numpy as np
from sklearn.datasets import load_iris
from ..Basic_Algorithms import fcm_code
# from read_uci.dataset import fetch_data_from_local
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class num:
    def __init__(self, n1, n2):
        self.n1=n1
        self.n2=n2

def sortt(ar1, ar2):
    list1=[]
    for i in range(len(ar1)):
        list1.append(num(ar1[i], ar2[i]))
    result=sorted(list1, key=lambda obj: obj.n1 )
    return np.array(list(obj.n2 for obj in result))

class it2fcm:
    def __init__(self, max_iter, m1, m2, x, init_centroid, number_of_centroid):
        self.x=x
        self.max_iter=max_iter
        self.m1=m1
        self.m2=m2
        self.v=init_centroid
        self.number_of_centroid=number_of_centroid
        self.u=np.zeros(shape=[2, len(x), number_of_centroid])


    def compute_u_1(self, distance, m):
        return 1/np.sum((distance.T[:, :, np.newaxis]/distance)**(2/(m-1)), axis=2).T

        
    def compute_u_2(self): 
        distance=np.linalg.norm(self.x[:, np.newaxis, :]-self.v, axis=2)
        u1=self.compute_u_1(distance=distance, m=self.m1)
        u2=self.compute_u_1(distance=distance, m=self.m2)
        u1_u2=np.stack([u1, u2], axis=0)
        moc_so_sanh=1/np.sum((distance.T[:, :, np.newaxis]/distance), axis=2).T
        for i in range(len(self.x)):
            for j in range(self.number_of_centroid):
                if moc_so_sanh[i][j]<1/self.number_of_centroid:
                    tmp=u1_u2[0][i][j]
                    u1_u2[0][i][j]=u1_u2[1][i][j]
                    u1_u2[1][i][j]=tmp
        self.u=u1_u2
        # print(np.where(u1_u2[0]<u1_u2[1]))
        # exit()


    def compute_v(self):
        u_trung_binh=np.sum(self.u, axis=0)/2
        return np.sum(u_trung_binh.T[:, :, np.newaxis]*self.x, axis=1)/np.sum(u_trung_binh.T, axis=1, keepdims=True)


    def compute_c(self, lower_index, u_lower, upper_index, u_upper):
        c2=np.sum(self.x[lower_index]*u_lower[:, np.newaxis], axis=0) + np.sum(self.x[upper_index]*u_upper[:, np.newaxis], axis=0) 
        return c2/(np.sum(u_upper)+np.sum(u_lower))
    

    def sort_data(self):
        data=self.x.T
        list_index=[]
        for i in range(len(data)):
            tmp=np.argsort(data[i])
            list_index.append(tmp)
            data[i]=data[i][tmp]
        self.data=data
        self.list_index=np.stack(list_index, axis=0)



    def karnik_algo(self, c, clus, mode):
        for  z in range(self.max_iter):
            list_k=[]
            for i in range(len(self.data)):
                for j in range(len(self.data[i])):
                    if self.data[i][j] >= c[clus][i] or j==len(self.data[i])-1:
                        list_k.append(j)
                        break
            for i in range(len(list_k)):
                if list_k[i]==0:
                    list_k[i]+=1
                elif list_k[i]==len(self.data[i])-1:
                    list_k[i]-=1
            c_new=[]
            u_2=[]
            for i in range(len(list_k)):
                lower_index=self.list_index[i][:list_k[i]+1]
                upper_index=self.list_index[i][list_k[i]+1:]
                if mode==1:
                    u_lower=self.u[:, lower_index, clus][0]
                    u_upper=self.u[:, upper_index, clus][1]
                else:
                    u_lower=self.u[:, lower_index, clus][1]
                    u_upper=self.u[:, upper_index, clus][0]
                c_new.append(self.compute_c(lower_index=lower_index, u_lower=u_lower, upper_index=upper_index, u_upper=u_upper))
                u_2.append(sortt(self.list_index[i],np.concatenate([u_lower, u_upper])))
            c_new=np.sum(np.array(c_new), axis=0)/len(list_k)
            if np.linalg.norm(c_new-c[clus]) < 1e-2:
                return c_new, np.sum(np.stack(u_2, axis=0).T, axis=1)/len(self.x[0])
            c[clus]=c_new
        return c_new, np.sum(np.stack(u_2, axis=0).T, axis=1)/len(self.x[0])
        
            

                


    def run(self):
        self.sort_data()
        # self.compute_u_2()
        # c=self.compute_v()
        for i in range(self.max_iter):
            self.compute_u_2()
            c=self.compute_v()
            upper_centroid=[]
            lower_centroid=[]
            real_u=[]
            
            for j in range(self.number_of_centroid): 
                km_for_upper=self.karnik_algo(c, j, 1)
                upper_centroid.append(km_for_upper[0])
                km_for_lower=self.karnik_algo(c, j, 0)
                lower_centroid.append(km_for_lower[0])
                real_u.append((km_for_upper[1]+km_for_lower[1])/2)
            upper_centroid=np.array(upper_centroid)
            lower_centroid=np.array(lower_centroid)

            hard_partioning=(upper_centroid+lower_centroid)/2

            self.u=np.stack(real_u, axis=0).T
            if np.linalg.norm(self.v-hard_partioning)<1e-2:
                self.v=hard_partioning
                return np.stack(real_u, axis=0).T
            self.v=hard_partioning
        


if __name__ == '__main__':
    import pandas as pd


    data=load_iris()
    x=data['data']
    y=data['target']
    init_centroid=np.array([[4.8, 3, 1, 0.3],[4, 2, 3, 1],[6, 3, 6, 1]])

   

    # tmp=pd.read_excel("Dry_Bean_Dataset.xlsx")
    # x, y=np.array(tmp.iloc[:, 0:16]), pd.factorize(np.array(tmp.iloc[:,16]))[0]
    # init_centroid=np.array([[28395,	610.29,	208.18,	173.89,	1.20, 0.55,	28715, 190.14, 0.76, 0.99, 0.96, 0.91, 0.01, 0.00, 0.83, 1.00],
    #                         [80340,	1105.49, 428.20, 240.56, 1.78, 0.83, 81292,	319.83,	0.75, 0.99,	0.83, 0.75,	0.01, 0.00,	0.56, 0.99],
    #                         [40704,	788.39,	318.93,	163.66,	1.95, 0.86,	41333, 227.65, 0.64, 0.98, 0.82, 0.71, 0.01, 0.00, 0.51, 0.99],
    #                         [52150,	912.21,	378.85,	175.53,	2.16, 0.89,	52747,257.68, 0.61, 0.99, 0.79, 0.68, 0.01, 0.00, 0.46, 1.00],
    #                         [61491,	1006.54, 399.59, 198.57, 2.01, 0.87, 62785,	279.81,	0.75, 0.98,	0.76, 0.70,	0.01, 0.00,	0.49, 0.99],
    #                         [41635,	765.73,	289.96,	183.38,	1.58, 0.77,	42018, 230.24, 0.69, 0.99, 0.89, 0.79, 0.01, 0.00, 0.63, 1.00],
    #                         [38162,	713.93,	266.69,	182.64,	1.46, 0.73,	38435, 220.43, 0.82, 0.99, 0.94, 0.83, 0.01, 0.00, 0.68, 1.00]
    #                         ])
    # fcmm=fcm_code.fcm(eps=1e-2, max_iter=1000, m=2)
    # u, v, i=fcmm.start(x, 7)

        

    # fcmm=fcm_code.fcm(eps=1e-2, max_iter=1000, m=2)
    # u, v, i=fcmm.start(x, 7, init_centroid)
    # print("\n\nChi so FCM")
    # fcm_code.validity2(data=x, clustter=v, membership=u, target=y)
    # print("\n\n")


    run_it2fcm=it2fcm(max_iter=1000, m1=2, m2=3, x=x, init_centroid=init_centroid, number_of_centroid=3)
    real_u=run_it2fcm.run()

    print('Chi so IT2FCM')
    fcm_code.validity2(data=x, clustter=run_it2fcm.v, membership=real_u, target=y)
    print("\n\n")
