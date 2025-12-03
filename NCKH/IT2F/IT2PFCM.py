import numpy as np
from sklearn.datasets import load_iris, load_wine
from ssfcm_code import ssfcm
import validity
import pandas as pd
from image_process import image_pr
from fcm import fcm
from s2cfc_code import s2cfcm, ln, exp
from IT2FCM_Thanh import it2fcm, num, sortt
from pcm import pcm

class it2pfcm:

    def __init__(self, data, max_iter, num_of_clus, init_typical, init_centroid, init_membership, eps=1e-4, m1=2, m2=3, a=1):
        self.data, self.max_iter=data, max_iter
        self.num_of_clus, self.eps, self.m1, self.m2=num_of_clus, eps, m1, m2
        self.t, self.v=init_typical, init_centroid
        self.u, self.a=init_membership, a
        

    def caculate_gamma(self, distance, m):
        return np.sum((self.a*self.u**m + self.t**m) * distance**2, axis=0)/np.sum((self.a*self.u**m + self.t**m), axis=0)


    def compute_u_1(self, m, distance):
        tu=distance
        mau=tu.T[:, : , np.newaxis]
        return 1/np.sum((tu/mau) ** (2/(m-1)), axis=0)
    

    def compute_u_2(self, distance):
        u1=self.compute_u_1(m=self.m1, distance=distance)
        u2=self.compute_u_1(m=self.m2, distance=distance)
        u1_u2=np.stack([u1, u2], axis=0)
        index=np.where(u1_u2[0]>=u1_u2[1])
        tmp=u1_u2[0][index]
        u1_u2[0][index]=u1_u2[1][index]
        u1_u2[1][index]=tmp
        return u1_u2
    


    def compute_t_1(self, distance, m, gamma):
        return 1/(1+((distance**2)/gamma)**(1/(m-1)))
    

    def compute_t_2(self, distance, gamma1, gamma2):
        t1=self.compute_t_1(distance=distance, m=self.m1, gamma=gamma1)
        t2=self.compute_t_1(distance=distance, m=self.m2, gamma=gamma2)
        t1_t2=np.stack([t1, t2], axis=0)
        index=np.where(t1_t2[0]>=t1_t2[1])
        tmp=t1_t2[0][index]
        t1_t2[0][index]=t1_t2[1][index]
        t1_t2[1][index]=tmp
        return t1_t2
    

    def sort_data(self):
        data=self.data.T
        list_index=[]
        for j in range(len(data)):
            tmp=np.argsort(data[j])
            list_index.append(tmp)
            data[j]=data[j][tmp]
        self.data=data.T
        return np.stack(list_index, axis=0)



    def compute_c(self, lower_index, u_lower, upper_index, u_upper, t_lower, t_upper):
        c2=np.sum(self.data[lower_index]*(self.a*u_lower[:, np.newaxis]+t_lower[:, np.newaxis]), axis=0) + np.sum(self.data[upper_index]*(self.a*u_upper[:, np.newaxis]+t_upper[:, np.newaxis]), axis=0) 
        return c2/(np.sum(self.a*u_upper+t_upper)+np.sum(self.a*u_lower+t_lower))


    def compute_v(self, u1_u2, t1_t2):
        utb, ttb=np.sum(u1_u2, axis=0)/2, np.sum(t1_t2, axis=0)/2
        u_pw_m= self.a*(utb.T ** self.m1) + (ttb.T ** self.m1)
        tu=np.dot(u_pw_m, self.data)
        mau=np.sum(u_pw_m, axis=1, keepdims=True)
        return tu/mau


    def karnik_algo(self, v, clus, mode, u1_u2, t1_t2):
        data=self.data.T
        for  z in range(self.max_iter):
            list_k=[]
            for i in range(len(data)):
                for j in range(len(data[i])):
                    if data[i][j] >= v[clus][i] or j==len(data[i])-1:
                        list_k.append(j)
                        break
            for i in range(len(list_k)):
                if list_k[i]==0:
                    list_k[i]+=1
                elif list_k[i]==len(data[i])-1:
                    list_k[i]-=1
            v_new=[]
            u_2=[]
            for i in range(len(list_k)):
                lower_index=self.list_index[i][:list_k[i]+1]
                upper_index=self.list_index[i][list_k[i]+1:]
                if mode==1:
                    u_lower=u1_u2[:, lower_index, clus][0]
                    t_lower=t1_t2[:, lower_index, clus][0]
                    u_upper=u1_u2[:, upper_index, clus][1]
                    t_upper=t1_t2[:, upper_index, clus][1]
                else:
                    u_lower=u1_u2[:, lower_index, clus][1]
                    t_lower=t1_t2[:, lower_index, clus][1]
                    u_upper=u1_u2[:, upper_index, clus][0]
                    t_upper=t1_t2[:, upper_index, clus][0]


                v_new.append(self.compute_c(lower_index=lower_index, u_lower=u_lower, 
                                            upper_index=upper_index, u_upper=u_upper, 
                                            t_lower=t_lower, t_upper=t_upper))
                u_2.append(sortt(self.list_index[i], np.concatenate([u_lower, u_upper])))
            v_new=np.sum(np.array(v_new), axis=0)/len(list_k)
            if np.linalg.norm(v_new-v[clus]) < 1e-2:
                return v_new, np.sum(np.stack(u_2, axis=0).T, axis=1)/len(self.data[0])
            v[clus]=v_new
        return v_new, np.sum(np.stack(u_2, axis=0).T, axis=1)/len(self.data[0])
        

    


        


    # def compute_v(self, u1_u2:np.ndarray):
    #     u_trung_binh=np.sum(u1_u2, axis=0)/2
    #     u_pw_m= self.a*(u_trung_binh.T ** self.m) + (self.t.T ** self.n)
    #     tu=np.dot(u_pw_m, self.data)
    #     mau=np.sum(u_pw_m, axis=1, keepdims=True)
    #     return tu/mau
    
    def min_max(self, distance):
        min=np.min(distance)
        max=np.max(distance)
        return 0.01+(distance-min)/(max-min)
    

    def fit(self):
        self.list_index=self.sort_data()
        for i in range(self.max_iter):
            print(i)
            distance=self.min_max(np.linalg.norm(self.data[:, np.newaxis, :]-self.v, axis=2))

            gamma_1=self.caculate_gamma(distance=distance, m=self.m1)
            gamma_2=self.caculate_gamma(distance=distance, m=self.m2)
            u1_u2=self.compute_u_2(distance=distance)
            t1_t2=self.compute_t_2(distance=distance, gamma1=gamma_1, gamma2=gamma_2)
            v_trung_binh=self.compute_v(u1_u2=u1_u2, t1_t2=t1_t2)

            upper_centroid=[]
            lower_centroid=[]
            real_u=[]

            for h in range(self.num_of_clus):
                km_for_upper=self.karnik_algo(v=v_trung_binh, clus=h, mode=1, u1_u2=u1_u2, t1_t2=t1_t2)
                upper_centroid.append(km_for_upper[0])

                km_for_lower=self.karnik_algo(v=v_trung_binh, clus=h, mode=0, u1_u2=u1_u2, t1_t2=t1_t2)
                lower_centroid.append(km_for_lower[0])

                real_u.append((km_for_upper[1]+km_for_lower[1])/2)
            upper_centroid=np.array(upper_centroid)
            lower_centroid=np.array(lower_centroid)

            hard_partioning=(upper_centroid+lower_centroid)/2
            self.u=np.stack(real_u, axis=0).T
            if np.linalg.norm(self.v-hard_partioning)<self.eps:
                self.v=hard_partioning
                break
            self.v=hard_partioning

if __name__== '__main__':

    np.random.seed(42)
    data_origin=load_iris()
    data=data_origin['data']
    target=data_origin['target']

    fcm_object=fcm(data=data, max_iter=1000, num_of_clus=3, eps=1e-3, m=2)
    fcm_object.fit()
    pcm_object=pcm(data=data, max_iter=1000, num_of_clus=3, eps=1e-3, m=2, init_typical=fcm_object.u, init_centroid=fcm_object.v)
    pcm_object.fit()
    pfcm_object=it2pfcm(data=data, max_iter=100, num_of_clus=3, eps=1e-3,
                        m1=2, m2=3, init_typical=pcm_object.t, init_centroid=pcm_object.v, 
                        init_membership=fcm_object.u, a=10)
    pfcm_object.fit()


    pfcm_object.u[:, [0, 2]]=pfcm_object.u[:, [2, 0]]


    from s2cfc_code import validity2
    print('IT2PFCM')
    validity2(data=[data], clustter=[pfcm_object.v], membership=[pfcm_object.u], target=[target])
    print()
    print('FCM')
    fcm_object.u[:, [0, 2]]=fcm_object.u[:, [2, 0]]
    validity2(data=[data], clustter=[fcm_object.v], membership=[fcm_object.u], target=[target])
    print()


