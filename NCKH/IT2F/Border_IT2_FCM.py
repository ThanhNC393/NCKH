import numpy as np
from sklearn.datasets import load_iris
from ..Basic_Algorithms.fcm_code import fcm
from ..read_uci.dataset import fetch_data_from_local
import pandas as pd
from ..Validity import validity
from ..Process.image_process import image_pr


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

    #Hàm tính u trên và u dưới
    def compute_u(self, v): 
        distance=np.linalg.norm(self.x[:, np.newaxis, :]-v, axis=2)
        u1=1/np.sum((distance.T[:, :, np.newaxis]/distance)**(2/(self.m1-1)), axis=2).T
        u2=1/np.sum((distance.T[:, :, np.newaxis]/distance)**(2/(self.m2-1)), axis=2).T
        u1_u2=np.stack([u1, u2], axis=0)
        moc_so_sanh=1/np.sum((distance.T[:, :, np.newaxis]/distance), axis=2).T
        for i in range(len(self.x)):
            for j in range(self.number_of_centroid):
                if moc_so_sanh[i][j]<1/self.number_of_centroid:
                    tmp=u1_u2[0][i][j]
                    u1_u2[0][i][j]=u1_u2[1][i][j]
                    u1_u2[1][i][j]=tmp
        self.u=u1_u2


    #Hàm tính v
    def compute_v(self):
        u_trung_binh=np.sum(self.u, axis=0)/2
        return np.sum(u_trung_binh.T[:, :, np.newaxis]*self.x, axis=1)/np.sum(u_trung_binh.T, axis=1, keepdims=True)


    #Hàm tính c cho thuật toán Karnik
    def compute_c(self, lower_index, u_lower, upper_index, u_upper):
        c2=np.sum(self.x[lower_index]*u_lower[:, np.newaxis], axis=0) + np.sum(self.x[upper_index]*u_upper[:, np.newaxis], axis=0) 
        return c2/(np.sum(u_upper)+np.sum(u_lower))


    #Hàm tính lại danh sách n (danh sách số lượng dữ liệu trong mỗi điểm dữ)
    def count_list_n(self, data: np.ndarray, v:np.ndarray):
        distance=np.abs(data - v[:, :, np.newaxis])
        datas_clus=np.argmin(distance, axis=0)
        labels=np.arange(self.number_of_centroid)
        count=[]
        for i in range(len(datas_clus)):
            tmp=[]
            for j in labels:
                tmp.append(len(np.where(datas_clus[i]==j)[0]))
            count.append(tmp)
        return count



    #Hàm chạy state 1 cho thuật toán border
    def state_1_for_border(self, data: np.ndarray):
        initial_centroid=data[np.arange(0, len(data), len(data)/self.number_of_centroid).astype(int)]
        cluster=list([[] for i in range(self.number_of_centroid)])
        for i in range(len(data)):
            cluster[np.argmin(np.abs(data[i]-initial_centroid))].append(data[i])
        n=[len(cluster[i]) for i in range(len(cluster))]
        for i in range(len(cluster)):
            cluster[i]=np.sum(cluster[i])/len(cluster[i])
        return n, np.array(cluster)


    #Hàm chạy state cho thuật toán border
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
    
    #Hàm chạy thuật toán Karnik để tìm v trên v dưới tối ưu
    def karnik_algo(self, clus, data, c, list_index, mode):
        for  z in range(100):
            list_k=[]
            for i in range(len(data)):
                for j in range(len(data[i])):
                    if data[i][j] >= c[clus][i] or j==len(data[i])-1:
                        list_k.append(j)
                        break
            for i in range(len(list_k)):
                if list_k[i]==0:
                    list_k[i]+=1
                elif list_k[i]==len(data[i])-1:
                    list_k[i]-=1
            c_new=[]
            u_2=[]
            for i in range(len(list_k)):
                lower_index=list_index[i][:list_k[i]+1]
                upper_index=list_index[i][list_k[i]+1:]
                if mode==1:
                    u_lower=self.u[:, lower_index, clus][0]
                    u_upper=self.u[:, upper_index, clus][1]
                else:
                    u_lower=self.u[:, lower_index, clus][1]
                    u_upper=self.u[:, upper_index, clus][0]
                c_new.append(self.compute_c(lower_index=lower_index, u_lower=u_lower, upper_index=upper_index, u_upper=u_upper))
                u_2.append(sortt(list_index[i], np.concatenate([u_lower, u_upper])))
            c_new=np.sum(np.array(c_new), axis=0)/len(list_k)
            if np.linalg.norm(c_new-c[clus]) < 1 :
                return c_new, np.sum(np.stack(u_2, axis=0).T, axis=1)/len(self.x[0])
            c[clus]=c_new
        return c_new, np.sum(np.stack(u_2, axis=0).T, axis=1)/len(self.x[0])
        
            


    def run(self):

        #1.Chạy state 1 cho từng chiều dữ liệu
        self.n=[[] for i in range(len(self.x[0]))]
        list_index=[]
        data=self.x.T
        v2=[]
        for j in range(len(data)):
            tmp=np.argsort(data[j])
            list_index.append(tmp)
            data[j]=data[j][tmp]
            self.n[j], v = self.state_1_for_border(data=data[j])
            v2.append(v)
        self.v=np.array(v2).T


        #2.Bắt đầu vòng lặp lớn
        for i in range(self.max_iter):
            list_v=[[] for i2 in range(len(data)) ]


            #3.Chạy state 2 cho từng chiều dữ liệu
            for i2 in range(len(data)):
                list_v[i2].append( self.state_2_for_border(data=data[i2], cluster=self.v.T[i2].copy(), n=self.n[i2]))
            list_v=np.array([np.array(*list_v[l]) for l in range(len(list_v))])
            list_index=np.stack(list_index, axis=0)


            #4.Chạy thuật toán it2cfc
            self.compute_u(list_v.T)
            upper_centroid=[]
            lower_centroid=[]
            real_u=[]
            c=self.compute_v()
            for j in range(self.number_of_centroid):
                km_for_upper=self.karnik_algo(j, data=data, c=c, list_index=list_index,mode= 1)
                upper_centroid.append(km_for_upper[0])
                km_for_lower=self.karnik_algo(j, data=data, c=c, list_index=list_index,mode= 0)
                lower_centroid.append(km_for_lower[0])
                real_u.append((km_for_upper[1]+km_for_lower[1])/2)
            upper_centroid=np.array(upper_centroid)
            lower_centroid=np.array(lower_centroid)
            hard_partioning=(upper_centroid+lower_centroid)/2
            hard_partioning=np.sort(hard_partioning, axis=0)
            if np.linalg.norm(self.v-hard_partioning)<1e-2:
                self.v=hard_partioning
                print(f'So lan lap la: {i}')
                return np.stack(real_u, axis=0).T
            self.v=hard_partioning


            #5. Tính toán lại danh sách n phục vụ cho lần lặp border sau
            self.n=self.count_list_n(data=data, v=self.v)


        print(f'So lan lap la: {i}')
        return np.stack(real_u, axis=0).T

def validity2( data: np.ndarray, clustter: np.ndarray, membership: np.ndarray, target: np.ndarray):
    tmp=pd.DataFrame(columns=["DB", "PC", "CE", "S", "CH", "SI", "FHV", "CS", "AC"])
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
        
        
def run():
    data=load_iris()
    x=data['data']
    y=data['target']
    init_centroid=np.array([[4.8, 3, 1, 0.3],[4, 2, 3, 1],[6, 3, 6, 1]])

    run_it2fcm=it2fcm(max_iter=1000, m1=2, m2=3, x=x, init_centroid=init_centroid, number_of_centroid=3)
    real_u=run_it2fcm.run()
    print('Chi so IT2FCM')
    validity2(data=x, clustter=run_it2fcm.v, membership=real_u, target=y)
    
    print("\n\n")


def run_image():
    np.random.seed(42)
    imgpr=image_pr(['NCKH/Data/b1_1024x1024.tif', 'NCKH/Data/b2_1024x1024.tif', 'NCKH/Data/b3_1024x1024.tif', 'NCKH/Data/b4_1024x1024.tif'])
    color=np.array([[0, 128, 0, 255],[128, 128, 128, 255],[0, 255, 0, 255],[1, 192, 255, 255],[0, 0, 255, 255],[0, 64, 0, 255]])
    data=imgpr.read_image()
    fcmrun=fcm(1000, 1e-3, 2)
    fcmrun.start(data= data, num_of_clus=6)
    print('Bắt đàu phân cụm ...')
    it2s2cfc_object=it2fcm(max_iter=1000, m1 = 2, m2 = 3, x=data, number_of_centroid=6, init_centroid=fcmrun.v)
    it2s2cfc_object.run()
    print('Phân cụm kết thúc!')
    imgpr.process(it2s2cfc_object.u, it2s2cfc_object.v, 1, 'hanoi_s2cfc_check.tif', color=color)




if __name__ == '__main__':
    pass

    



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


    
    # run_it2fcm=it2fcm(max_iter=30, m1=2, m2=3, x=x, init_centroid=init_centroid, number_of_centroid=7)
    # print('Chi so IT2FCM')

    # real_u=run_it2fcm.run()

    # fcm_code.validity2(data=x, clustter=run_it2fcm.v, membership=real_u, target=y)
    



        

    # fcmm=fcm_code.fcm(eps=1e-2, max_iter=1000, m=2)
    # u, v, i=fcmm.start(x, 3, init_centroid)
    # print("\nChi so FCM")
    # fcm_code.validity2(data=x, clustter=v, membership=u, target=y)
    # print("\n\n")



    

    