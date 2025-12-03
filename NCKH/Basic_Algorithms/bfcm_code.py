import numpy as np
import pandas as pd
from ..Validity import validity
from sklearn.datasets import load_iris
from ..Process.image_process import image_pr

class fcm():
    def __init__(self, max_iter, eps, m, num_of_clus):
        self.max_iter=max_iter
        self.eps=eps
        self.m=m
        self.num_of_clus= num_of_clus


    def state_1_for_border(self, data: np.ndarray):
        initial_centroid=data[np.arange(0, len(data), len(data)/self.num_of_clus).astype(int)]
        cluster=list([[] for i in range(self.num_of_clus)])
        for i in range(len(data)):
            cluster[np.argmin(np.abs(data[i]-initial_centroid))].append(data[i])
        n=[len(cluster[i]) for i in range(len(cluster))]
        for i in range(len(cluster)):
            cluster[i]=np.sum(cluster[i])/len(cluster[i])
        return n, np.array(cluster)


    def state_2_for_border(self, data, cluster, n):
        n_use=np.array([0]+n)
        cluster_use=np.insert(cluster, 0, 0)
        n_use[1]-=1
        for i in range(1, len(n_use)):
            n_use[i]=n_use[i]+n_use[i-1]
        for i in range(1, len(cluster_use)-1):
            border=(cluster_use[i]+cluster_use[i+1])/2
            while data[n_use[i]]>=border:
                n[i-1]-=1
                n[i]+=1
                n_use[i]-=1
            while data[n_use[i]+1]<border:
                n[i-1]+=1
                n[i]-=1
                n_use[i]+=1
        n_use=np.array([0]+n)
        for i in range(1, len(n_use)):
            n_use[i]=n_use[i]+n_use[i-1]
        for i in range(len(n_use)-1):
            data_of_clus=data[n_use[i]:n_use[i+1]]
            cluster[i]=np.sum(data_of_clus)/len(data_of_clus)
        return cluster


    def update_v(self, u: np.ndarray, data: np.ndarray):

        u_pw_m=u.T ** self.m
        tu=np.dot(u_pw_m, data)
        mau=np.sum(u_pw_m, axis=1, keepdims=True)
        return tu/mau


    def update_u(self, data: np.ndarray, v: np.ndarray):

        tu=np.linalg.norm(data[:, np.newaxis, :]-v, axis=2)
        mau=tu.T[:, : , np.newaxis]
        return 1/np.sum((tu/mau) ** (2/(self.m-1)), axis=0)

    def count_list_n(self, data: np.ndarray, v:np.ndarray):
        distance=np.abs(data - v[:, :, np.newaxis])
        datas_clus=np.argmin(distance, axis=0)
        labels=np.arange(self.num_of_clus)
        count=[]
        for i in range(len(datas_clus)):
            tmp=[]
            for j in labels:
                tmp.append(len(np.where(datas_clus[i]==j)[0]))
            count.append(tmp)
        return count


    def start(self, data: np.ndarray, init_centroid=None):
        self.n=[[] for i in range(len(data[0]))]
        self.v=[[] for i in range(len(data[0]))]

        self.data=data.T
        for i in range(len(self.data)):
            self.n[i], self.v[i] = self.state_1_for_border(data=self.data[i])
        self.v=np.array(self.v).T

        for i in range(self.max_iter):
            v=[[] for i2 in range(len(self.data)) ]

            for j in range(len(self.data)):
                v[j].append( self.state_2_for_border(data=self.data[j], cluster=self.v.T[j].copy(), n=self.n[j])[0])
            self.v=np.array(self.v)
            if i>0:
                u_old=self.u.copy()

            self.u=self.update_u(data, self.v)
            self.v=self.update_v(self.u, data)
            if i>0:
                tmp2=np.linalg.norm(u_old-self.u, ord=2)
                print(i, tmp2)
                if tmp2<self.eps:
                    return self.u, self.v, i
                self.n=self.count_list_n(data=self.data, v=self.v)
            
        
       
        
        # for i in range(self.max_iter):
        #     old_u=u.copy()
        #     u=self.update_u(data, v)
        #     v=self.update_v(u, data)

        #     tmp2=np.linalg.norm(u-old_u, ord=2)
        #     print(i, tmp2)
        #     if tmp2<self.eps:
        #         self.u=u
        #         self.v=v 
        #         return u, v, i

        # return u,v,i
    
def validity2( data: np.ndarray, clustter: np.ndarray, membership: np.ndarray, target: np.ndarray):
    tmp=pd.DataFrame(columns=["DB-", "PC+", "CE-", "S+", "CH+", "SI", "FHV", "CS+", "AC"])
    tmp2=[]
    tmp2.append(validity.davies_bouldin(data, np.argmax(membership, axis=1)))
    tmp2.append(validity.partition_coefficient(membership))
    tmp2.append(validity.classification_entropy(membership))
    tmp2.append(validity.separation(data, membership, clustter))
    tmp2.append(validity.calinski_harabasz(data, np.argmax(membership, axis=1)))
    tmp2.append(validity.silhouette(data, np.argmax(membership, axis=1)))
    tmp2.append(validity.hypervolume(membership))
    tmp2.append(validity.cs(data, membership, clustter))
    tmp2.append(validity.accuracy_score(target, np.argmax(membership, axis=1)))
    tmp.loc[len(tmp)]=tmp2
    print(tmp)

if __name__=='__main__':

    tmp=pd.read_excel("Dry_Bean_Dataset.xlsx")
    data, target=np.array(tmp.iloc[:, 0:16]), pd.factorize(np.array(tmp.iloc[:,16]))[0]
    rand_index=np.random.choice(np.arange(len(data)), size=len(data), replace=False)
    data, target=data[rand_index], target[rand_index]
    fcm1=fcm(1000, 1e-4, 2, 7)
    u, v, i=fcm1.start(data=data)
    print('\n\n')
    validity2(data, v, u, target=target)
    print('\n\n')


    # data=load_iris()
    # x=data['data']
    # y=data['target']
    # fcm1=fcm(1000, 1e-4, 2, 3)
    # u, v, i=fcm1.start(data=x)
    # print('\nChi so FCM\n')
    # validity2(x, v, u, target=y)
    # print('\n\n')
