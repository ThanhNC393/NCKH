from .fcm_code import fcm, validity2
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_iris
from ..Process.image_process import image_pr



class ssfcm(fcm):
    def __init__(self, max_iter, eps, m):
        super().__init__(max_iter, eps, m)
        np.random.seed(42)

    def init_semi_sup_data(self, num_of_clus: int, per: int, target: np.ndarray):
        length=len(target)
        self.u_bar=np.zeros([length, num_of_clus])
        rand_index_data=np.random.choice(np.arange(length), size=int((length/100)*per), replace=False)
        self.u_bar[rand_index_data, target[rand_index_data]]=1
        self.l=np.zeros(length)
        self.l[rand_index_data]=1

    def update_u(self, data: np.ndarray, v: np.ndarray):
        u_bar_minus=1-np.sum(self.u_bar, axis=1, keepdims=True)
        dki=1/np.power(np.linalg.norm(data[:, np.newaxis, :]-v, axis=2),2/(self.m-1))
        return self.u_bar + u_bar_minus*(dki/(np.sum(dki, axis=1,keepdims=True)))

    def update_v(self, u: np.ndarray, data: np.ndarray):
        u_pw_m=(u - self.u_bar).T ** self.m 
        tu=np.dot(u_pw_m, data)
        mau=np.sum(u_pw_m, axis=1, keepdims=True)
        mau[np.where(np.abs(mau)<1e-6)[0].tolist()]=1 
        return tu/mau
    
    def start(self, data: np.ndarray, num_of_clus: int, per: int, target):
        self.init_semi_sup_data(num_of_clus=num_of_clus, per=per, target=target)
        return super().start(data, num_of_clus)

if __name__=='__main__':

    np.random.seed(42)
    
    imgpr=image_pr(['b1_1024x1024.tif', 'b2_1024x1024.tif', 'b3_1024x1024.tif', 'b4_1024x1024.tif'])
    color=np.array([[0, 128, 0, 255],[128, 128, 128, 255],[0, 255, 0, 255],[1, 192, 255, 255],[0, 0, 255, 255],[0, 64, 0, 255]])
    data=imgpr.read_image()
    none_black=np.all(data!=[0,0,0,0], axis=1)
    index=np.where(none_black==True)[0].tolist()

    # fcmrun=fcm(1000, 1e-3, 2)
    # u, v, i=fcmrun.start(data=data[index], num_of_clus=6)
    fcm_run=ssfcm(1000, 1e-4, 2)
    fcmrun=fcm(1, 1e-3, 2)
    tmp=list(fcmrun.start(data=data[index], num_of_clus=6))[0]
    u, v, i=fcm_run.start(data=data[index], num_of_clus=6, per=30, target=np.argmax(list(fcmrun.start(data=data[index], num_of_clus=6))[0], axis=1))

    imgpr.process([u], [v], 1, 'hanoi_ssfcm_check.tif', index=index, color=color)

    # tmp=pd.read_excel("Dry_Bean_Dataset.xlsx")
    # data, target=np.array(tmp.iloc[:, 0:16]), pd.factorize(np.array(tmp.iloc[:,16]))[0]
    # rand_index=np.random.choice(np.arange(len(target)), size=len(target), replace=False)
    # data, target=data[rand_index], target[rand_index]

    # tmp=load_iris()
    # data, target = tmp['data'], tmp['target']
    # ssfcm_model=ssfcm(max_iter=1000, eps=1e-4, m=2)
    # u, v, i=ssfcm_model.start(data=data, num_of_clus=3, per=30, target=target)


    # print('\nSSFCM\n')
    # validity2(data, v, u, target)
    # print('\n\n')

