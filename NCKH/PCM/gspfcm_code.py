import numpy as np
from sklearn.datasets import load_iris
from fcm_code import fcm
import validity
import pandas as pd
from nhap2 import remove_label

class gspfcm():
    
    def __init__(self, a: float, b: float, eta: float, K:float = 1, m: float = 2, eps: float = 1e-5, gamma=1, maxiter: int = 10000):
        self.a, self.b, self.eta, self.gamma= a, b, eta, gamma
        self.K, self.m, self.eps, self.maxiter = K, m, eps, maxiter
        self.run_fcm=fcm(max_iter=1000, eps=1e-4, m=2)


    def init_semi_data(self, data: np.ndarray, label: np.ndarray, ratio_label: float):
        list_label = np.unique(label)

        # list_index=[]
        # for i in list_label:
        #     list_index.append(list(np.where(label==i)[0].tolist()))
        # rand_index=[]
        # for i in list_index:
        #     rand_index.append(np.random.choice(i, size=int((len(i)/100)*ratio_label), replace=False))
        # v_star=[]
        # for i in rand_index:
        #     v_star.append(np.mean(data[i.tolist()], axis=0))
        # self.v_star=np.stack(v_star, axis=0)
        # l=np.zeros(len(data), dtype=int)
        # l[np.concatenate(rand_index, axis=0)]=1
        # self.l=l

        index_a_manh=remove_label(labels=label, labeled_percentage=ratio_label/100)[0]
        l=[]
        v=list([] for i in range(len(list_label)))
        for i in range(len(index_a_manh)):
            if index_a_manh[i]!=-1: 
                l.append(1)
                v[index_a_manh[i]].append(i)
            else: l.append(0)
        for i in range(len(v)):
            v[i]=np.mean(data[v[i]], axis=0)
        self.v_star=np.stack(v, axis=0)
        self.l=l


    def caculate_u_star(self, distance: np.ndarray, shape):
        distance=np.power(distance, 2/(self.m-1))
        u_star=1/np.sum(distance[:, :, np.newaxis]/distance[:, np.newaxis, :], axis=2)
        tmp=np.zeros(shape)
        index=np.where(np.array(self.l)==1)[0].tolist()
        tmp[index]=u_star[index]
        self.u_star = tmp


    def caculate_t(self, u: np.ndarray, distance: np.ndarray, shape=None):
        minn, maxx=np.min(distance), np.max(distance)
        distance_cp=2*(distance-minn)/(maxx-minn)
        ray=self.K * np.sum(u ** self.eta * distance_cp ** 2, axis=0) / np.sum(u ** self.eta, axis=0)
        t = 1/(1+((self.b * distance_cp ** 2 )/ ray) ** (1/(self.eta-1)))
        if shape is not None:
            index=np.where(np.array(self.l)==1)[0].tolist()
            t2=np.zeros(shape)
            t2[index]=t[index]
            return t2
        return t
        

    def update_u(self, distance: np.ndarray):
        common=(1/distance**2)**(1/(self.m-1))
        tmp=1-np.sum(self.u_star, axis=1, keepdims=True)
        tmp[np.where(np.abs(tmp-1)<1e-5)[0].tolist()]=1
        return self.u_star + tmp*common/np.sum(common, axis=1, keepdims=True)


    def update_t(self, u, data):
        distance=np.linalg.norm(data[:, np.newaxis, :]-self.v, axis=2)+np.linalg.norm(self.v-self.v_star, axis=1)*self.gamma
        ki_hieu_la=self.K * np.sum(u ** self.eta * distance ** 2, axis=0) / np.sum(u ** self.eta, axis=0)
        common=(ki_hieu_la/(self.b*distance**2))**(1/(self.eta-1))
        gt=(self.t_star + common)/(1+common)
        lt=(self.t_star - common)/(1-common)
        index=np.where(self.t>=self.t_star)[0].tolist()
        lt[index]=gt[index]
        return lt
    

    def update_v(self, data):
        common=self.a*np.abs(self.u-self.u_star)**self.m + self.b*np.abs(self.t-self.t_star)**self.eta
        tu=np.sum((data + self.v_star[:, np.newaxis, :])*common.T[:, :, np.newaxis], axis=1)
        mau=np.sum(common*(self.gamma+1), axis=0)
        return tu/mau[:, np.newaxis]


    def check_iter(self, u_old, t_old):
        tmp=np.linalg.norm(self.u-u_old) + np.linalg.norm(self.t-t_old)
        if tmp<self.eps:
            return True
        return False



    def train(self, data: np.ndarray, label: np.ndarray, num_of_clus: int, ratio_label:float):
        self.init_semi_data(data=data, label=label, ratio_label=ratio_label)
        distance=np.linalg.norm(data[:, np.newaxis, :]-self.v_star, axis=2)
        self.caculate_u_star(distance=distance, shape=(len(data), num_of_clus))
        self.t_star=self.caculate_t(u=self.u_star, distance=distance, shape=(len(data), num_of_clus))

        fcm_local=self.run_fcm.start(data=data, num_of_clus=num_of_clus)
        self.u, self.v = fcm_local[0], fcm_local[1]
        distance=np.linalg.norm(data[:, np.newaxis, :]-self.v, axis=2)
        self.t=self.caculate_t(u=self.u, distance=distance)
        self.u=self.update_u(distance)

        for i in range(self.maxiter):
            distance=np.linalg.norm(data[:, np.newaxis, :]-self.v, axis=2)
            u_old=self.u.copy()
            t_old=self.t.copy()
            self.v=self.update_v(data=data)
            self.u=self.update_u(distance=distance)
            self.t=self.update_t(self.u, data=data)
            if self.check_iter(u_old=u_old, t_old=t_old):
                return i
        return i
    
    
def validity2(data: list, clustter: list, membership: list, target: list):
    tmp=pd.DataFrame(columns=["DI", "DB", "PC", "CE", "CH", "SI", "FHV", "CS", "S", "AC"])
    for i in range(len(list(data))):
        tmp2=[]
        tmp2.append(validity.dunn(data[i], np.argmax(membership[i], axis=1)))
        tmp2.append(validity.davies_bouldin(data[i], np.argmax(membership[i], axis=1)))
        tmp2.append(validity.partition_coefficient(membership[i]))
        tmp2.append(validity.classification_entropy(membership[i]))
        tmp2.append(validity.calinski_harabasz(data[i], np.argmax(membership[i], axis=1)))
        tmp2.append(validity.silhouette(data[i], np.argmax(membership[i], axis=1)))
        tmp2.append(validity.hypervolume(membership[i]))
        tmp2.append(validity.cs(data[i], membership[i], clustter[i]))
        tmp2.append(validity.separation(data[i], membership[i], clustter[i]))
        tmp2.append(validity.accuracy_score(target[i], np.argmax(membership[i], axis=1)))
        tmp.loc[len(tmp)]=tmp2
    print(tmp)
        




if __name__ == "__main__":
    
    tmp=load_iris()
    data=tmp['data']
    label=tmp['target']
    np.random.seed(42)
    
    tmp=load_iris()
    data=tmp['data']
    label=tmp['target']
    rungsp=gspfcm(a=1, b=1, eta=2)
    print(rungsp.train(data=data, label=label, num_of_clus=3, ratio_label=30))
    validity2(data=[data], clustter=[rungsp.v], membership=[rungsp.u], target=[label])
    






    
        