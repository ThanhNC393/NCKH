import numpy as np
from gspfcm_code import gspfcm
from sklearn.datasets import load_iris
import pandas as pd
import validity


def split_dataa(data: np.ndarray, label: np.ndarray, num_of_site: int):
    list_label=np.unique(label)
    label_spl=list(np.where(label==i)[0] for i in list_label)
    data2, label2 = [], []
    list_data=list([] for i in range(num_of_site))
    list_label=list([] for i in range(num_of_site))
    for i in range(len(label_spl)):
        tmp4=np.array_split(label_spl[i], num_of_site)
        for j in range(num_of_site):
            list_data[j].append(data[tmp4[j].tolist()])
            list_label[j].append(label[tmp4[j].tolist()])
    for i in range(num_of_site):
        data2.append(np.concatenate(list_data[i], axis=0))
        label2.append(np.concatenate(list_label[i], axis=0))
    return data2, label2

def check_ter(u_old: np.ndarray, u_new: np.ndarray, mode: int, epsilon):

        if mode==1:
            if np.linalg.norm(u_new-u_old)<epsilon: 
                return True
            return False
        
        if mode==2:
            sum=0
            for i in range(len(u_old)):
                sum+=np.linalg.norm(u_old[i]-u_new[i])
            if sum<epsilon: return True
            return False

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

class gscpfcm:
    def __init__(self, data: np.ndarray, label: np.ndarray, beta: float, a: float, b: float, delta: float, eta: float, num_of_site: int, num_of_clus: int, ratio_label: float) -> None:
        self.data, self.label, self.beta, self.a, self.b, self.ratio_label = data, label, beta, a, b, ratio_label
        self.delta, self.eta, self.num_of_site, self.num_of_clus = delta, eta, num_of_site, num_of_clus
        self.list_site=[]

    def split_data(self):
        self.list_data, self.list_label=split_dataa(self.data, self.label, self.num_of_site)


    def phase1(self):
        self.list_gsp=[]
        for i in range(self.num_of_site):
            tmp=gspfcm(a=self.a, b=self.b, eta=self.eta)
            tmp.train(data=self.list_data[i], label=self.list_label[i], num_of_clus=self.num_of_clus, ratio_label=self.ratio_label)
            self.list_gsp.append(tmp)


    def make_u_colab(self):
        list_u_colab=[]
        for i in range(self.num_of_site):
            list_tmp=[]
            for j in range(self.num_of_site):
                if j!=i:
                    dis_xv=np.linalg.norm(self.list_data[i][:, np.newaxis, :]-self.list_gsp[j].v, axis=2)
                    dis_xv=dis_xv[:, :, np.newaxis] / dis_xv[:, np.newaxis, :]
                    dis_xv=np.sum(dis_xv ** 2, axis=2)
                    list_tmp.append(dis_xv**(-1))
            list_u_colab.append(np.stack (list_tmp, axis=0))
        self.list_u_colab = list_u_colab


    def update_u(self, index: int, tmp2, distance, big_B):

        cp1=self.a*big_B + np.sum(self.beta * distance, axis=0)
        cp1=1/(np.sum(cp1[:, :, np.newaxis] / cp1[:, np.newaxis, :] , axis=2))

        cp2=(tmp2.u_star * self.a*big_B + np.sum(self.beta*self.list_u_colab[index]*distance, axis=0)) / (self.a * big_B + distance)

        cp3=1-np.sum(cp2, axis=1, keepdims=True)
        
        return cp1*cp3 + cp2
    

    def update_v(self, index: int, tmp2):

        cp1=self.a * (tmp2.u-tmp2.u_star) ** 2 + self.b * (tmp2.t - tmp2.t_star) ** 2

        cp2=self.beta * (tmp2.u-self.list_u_colab[index])**2

        cp3=np.sum(cp1.T[:, : , np.newaxis ]*(self.list_data[index]+self.delta*tmp2.v[:, np.newaxis, :]), axis=1)

        cp4=np.sum(np.sum(np.transpose(cp2, (0, 2, 1))[:, :, :, np.newaxis]*self.list_data[index], axis=0), axis=1)

        return (cp3+cp4)/(np.sum(cp1*(self.delta+1), axis=0)[:, np.newaxis] + np.sum(np.sum(cp2, axis=0), axis=0)[:, np.newaxis])


    def update_t(self, tmp2, distance):

        gamma=tmp2.K * np.sum(tmp2.u ** self.eta * distance ** 2, axis=0)/np.sum(tmp2.u ** self.eta, axis=0)

        cp1=(gamma/(self.b * (distance ** 2 + self.delta * np.linalg.norm(tmp2.v - tmp2.v_star, axis=1)**2 ))) ** (1/(self.eta-1))

        opearation=np.array(list(list(-1 if i < 0 else 1 for i in j) for j in tmp2.t-tmp2.t_star))

        return (tmp2.t_star + opearation*cp1) / (1 + opearation*cp1) 


    def phase2(self, max_iter):
        
        self.make_u_colab()

        for i in range(max_iter):
            list_u_old=list(z.u for z in self.list_gsp)
            for index in range(self.num_of_site):
                tmp2=self.list_gsp[index]
                for j in range(max_iter):
                    distance=np.linalg.norm(self.list_data[index][:, np.newaxis, :]-tmp2.v, axis=2)
                    big_B=distance ** 2 + self.delta * np.linalg.norm(tmp2.v - tmp2.v_star, axis=1) ** 2
                    minn, maxx = np.min(big_B), np.max(big_B)
                    big_B=(big_B-minn)/(maxx-minn)
                    tmp2.t=self.update_t(tmp2=tmp2, distance=distance)
                    u_old=tmp2.u.copy()
                    tmp2.u=self.update_u(index=index, tmp2=tmp2, distance=distance, big_B=big_B)
                    tmp2.v=self.update_v(index=index, tmp2=tmp2)
                    if check_ter(u_old=u_old, u_new=tmp2.u, mode=1, epsilon=1e-5):
                        break
            list_u_new=list(z.u for z in self.list_gsp)
            if check_ter(u_old=list_u_old, u_new=list_u_new, epsilon=1e-6, mode=2):
                return i


    def train(self, max_iter):
        self.split_data()
        self.phase1()
        list_u=list(i.u for i in tmp.list_gsp)
        list_v=list(i.v for i in tmp.list_gsp)
        print('Phase 1')
        validity2(data=tmp.list_data, clustter=list_v, membership=list_u, target=tmp.list_label)
        self.phase2(max_iter=max_iter)
        


if __name__ == '__main__':
    iris=load_iris()
    data, label = iris['data'], iris['target']
    np.random.seed(42)
    tmp=gscpfcm(data=data, label=label, beta=0.5, a=1, b=1, delta=1, eta=2, num_of_site=3, num_of_clus=3, ratio_label=30)
    tmp.train(max_iter=1000)
    list_u=list(i.u for i in tmp.list_gsp)
    list_v=list(i.v for i in tmp.list_gsp)
    print('Phase 2')
    validity2(data=tmp.list_data, clustter=list_v, membership=list_u, target=tmp.list_label)

        