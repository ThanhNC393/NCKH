import numpy as np
from sklearn.datasets import load_iris, load_wine
from ssfcm_code import ssfcm
import validity
import pandas as pd
from image_process import image_pr
from fcm_code import fcm
from s2cfc_code import s2cfcm, ln, exp
from IT2FCM_Thanh import it2fcm, num, sortt



class pcm():
    def __init__(self, data, max_iter, num_of_clus, init_typical, init_centroid, eps=1e-4, m=2):
        self.max_iter=max_iter
        self.eps=eps
        self.m=m
        self.data=data
        self.num_of_clus=num_of_clus
        self.t=init_typical
        self.v=init_centroid

    def normalize_distance(self, d):
        distance=np.linalg.norm(self.data[:, np.newaxis, :]-self.v, axis=2)
        min=np.min(distance)
        max=np.max(distance)
        self.distance=d*(distance-min)/(max-min)


    def caculate_gamma(self):
        return np.sum(self.t**2 * self.distance**2, axis=0)/np.sum(self.t**2, axis=0)


    def update_v(self):

        u_pw_m=self.t.T ** self.m
        tu=np.dot(u_pw_m, self.data)
        mau=np.sum(u_pw_m, axis=1, keepdims=True)
        return tu/mau


    def update_t(self):
        return 1/(1+(self.distance**2/self.gamma)**(1/(self.m-1)))



    def fit(self):
        self.normalize_distance(d=1)
        self.gamma=self.caculate_gamma()
        for i in range(self.max_iter):
            if i!=0:
                self.normalize_distance(d=2)
            old_v=self.v.copy()
            self.t=self.update_t()
            self.v=self.update_v()
            tmp2=np.linalg.norm(self.v-old_v)
            if tmp2<self.eps:
                return i
        return i
    

if __name__=='__main__':
    np.random.seed(42)
    data_origin=pd.read_csv('data/data2.csv')
    data, target=np.array(data_origin.iloc[:, 0:2]), pd.factorize(np.array(data_origin.iloc[:,2]))[0]
    fcm_object=fcm(data=data, max_iter=1000, num_of_clus=2, eps=1e-3, m=2)
    fcm_object.fit()
    pcm_object=pcm(data=data, max_iter=1000, num_of_clus=2, eps=1e-3, m=2, init_typical=fcm_object.u, init_centroid=fcm_object.v)
    pcm_object.fit()
    

    




    # np.random.seed(42)

    # imgpr=image_process.image_pr(['data/b1_1024x1024.tif', 'data/b2_1024x1024.tif', 'data/b3_1024x1024.tif', 'data/b4_1024x1024.tif'])
    # color=np.array([[0, 128, 0, 255],[128, 128, 128, 255],[0, 255, 0, 255],[1, 192, 255, 255],[0, 0, 255, 255],[0, 64, 0, 255]])
    # data=imgpr.read_image(mode=1)

    # fcm_object=fcm(data=data, max_iter=1000, num_of_clus=6, eps=1e-3, m=2)
    # fcm_object.fit()
    # pcm_object=pcm(data=data, max_iter=1000, num_of_clus=6, eps=1e-3, m=2, init_typical=fcm_object.u, init_centroid=fcm_object.v)
    # imgpr.process(list_u=[pcm_object.t], list_v=[pcm_object.v], num_of_data_site=1, name_output='nhappcm.png', color=color)
    
    