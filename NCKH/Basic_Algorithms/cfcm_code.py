import numpy as np
import pandas as pd
from ..Validity import validity
from .fcm_code import fcm
# from image_process import image_pr
from ..Process.image_process import image_pr
from sklearn.datasets import load_iris

class cfcm: 
    def __init__(self, max_iter, espsilon, beta: int, m: int, data: np.ndarray, num_of_data_site: int, num_of_clus: list, target=None):
        self.max_iter=max_iter
        self.espsilon=espsilon
        self.beta=beta
        self.m=m
        self.num_of_data_site=num_of_data_site
        self.num_of_clus=num_of_clus
        self.list_target=[]
        if target is None:
            self.list_data=np.array_split(data, num_of_data_site)
            # self.list_target=np.array_split(target, num_of_data_site)
        else:
            self.list_data=[[] for ii in range(num_of_data_site)]
            self.list_target=[[] for ii in range(num_of_data_site)]
            tmp1=np.unique(target)
            for i in tmp1:
                tmp4=np.where(target==i)[0].tolist()
                tmp4=np.array_split(tmp4, self.num_of_data_site)
                # tmp3=np.array_split(data[tmp4], num_of_data_site)
                for j in range(num_of_data_site):
                    self.list_data[j].append(data[tmp4[j]])
                    self.list_target[j].append(tmp4[j])
            self.list_data=[np.concatenate(self.list_data[ii], axis=0) for ii in range (self.num_of_data_site)]
            self.list_target=[np.concatenate(self.list_target[ii], axis=0) for ii in range (self.num_of_data_site)]
            # print(self.list_target)
        self.list_v=[]
        self.list_u=[]
        self.run_fcm=fcm(1000, espsilon, m)

        #Tạo u và v
        for i in range(len(self.list_data)):
            #Tạo v
            u, v=self.run_fcm.start(self.list_data[i], self.num_of_clus[i])[0:2]
            self.list_v.append(v)
            if target[0]==None:
                self.list_target.append(np.argmax(u, axis=1))
            #Tạo u
            tmp=np.random.rand(self.list_data[i].shape[0], self.num_of_clus[i])
            tmp/=np.sum(tmp, axis=1).reshape(-1,1)
            self.list_u.append(tmp)
    
    #Cập nhật u
    def update_u(self, v_nga: np.ndarray, i:int):
        x, y=self.list_data[i].shape
        a=self.list_v[i].shape[0]
        cp1=np.sum(np.power(self.list_data[i].reshape(x, 1, y)-self.list_v[i], 2), axis=2).T+self.beta*np.sum(np.power(self.list_v[i]-v_nga, 2), axis=1, keepdims=True)
        cp2=cp1.T.reshape(x, 1, a)
        cp3=cp1.T.reshape(x, a, 1)
        self.list_u[i]=1/np.sum(cp3/cp2, axis=2)

    #Cập nhật v
    def update_v(self, v_nga: np.ndarray, i: int):
        x, y=self.list_u[i].shape
        a, b=v_nga.shape
        u_power=np.power(self.list_u[i], self.m)
        cp1=np.sum(u_power.T.reshape(y, x, 1)*self.list_data[i], axis=1)
        cp2=np.sum(v_nga.reshape(a, b, 1)*u_power.T.reshape(y, 1, x), axis=2)*self.beta
        cp3=np.sum(u_power.T,axis=1, keepdims=True)*(self.beta+1)
        self.list_v[i]=(cp1+cp2)/(cp3)


    #Kiểm tra điều kiện dừng
    def check_ter(self, v_cu: np.ndarray, v_sau: np.ndarray):
        return np.linalg.norm(v_cu-v_sau)<self.espsilon
    

    def run(self):

        for i in range(self.max_iter):


            #Tạo v~
            print("Lần lặp thứ {}:".format(i+1))
            list_v_nga=[]
            check=dict([(i, -1) for i in set(self.num_of_clus)])
            for j in range(len(self.num_of_clus)):
                if check[self.num_of_clus[j]]==-1: 
                    list_v_nga.append(self.run_fcm.start(np.concatenate(self.list_v, axis=0), self.num_of_clus[j])[1]) 
                    check[self.num_of_clus[j]]=j
                else:
                    list_v_nga.append(list_v_nga[check[self.num_of_clus[j]]])


            #Giai đoanj cộng tác, tối ưu cho từng datasite
            list_v_cu=self.list_v.copy()
            for z in range(self.num_of_data_site):
                for j in range(1000):
                    v_cu2=self.list_v[z].copy()
                    self.update_u(list_v_nga[z], z)
                    self.update_v(list_v_nga[z], z)
                    if self.check_ter(v_cu2, self.list_v[z]): break
                print("\tdata site thứ {} hội tụ sau {} lần lặp".format(z+1, j+1))
            # for i in range(len(self.list_u)):
            #     print(np.argmax(self.list_u[i], axis=1))
            check=[]
            for t in range(self.num_of_data_site):
                check.append(self.check_ter(list_v_cu[t], self.list_v[t]))
            if all(check): return i
        return i

    #Chạy chỉ số
    # def validity2(self, data: list, clustter: list, membership: list, target: list):
    #     tmp=pd.DataFrame(columns=["DB", "PC", "CE", "S", "CH", "SI", "FHV", "CS", "F1", "AC"])
    #     for i in range(len(list(data))):
    #         tmp2=[]
    #         tmp2.append(validity.davies_bouldin_index(data[i], np.argmax(membership[i], axis=1)))
    #         tmp2.append(validity.partition_coefficient(membership[i]))
    #         tmp2.append(validity.classification_entropy(membership[i]))
    #         tmp2.append(validity.separation_index(data[i], membership[i], clustter[i]))
    #         tmp2.append(validity.calinski_harabasz_index(data[i], np.argmax(membership[i], axis=1)))
    #         tmp2.append(validity.silhouette_index(data[i], np.argmax(membership[i], axis=1)))
    #         tmp2.append(validity.fuzzy_hypervolume(membership[i]))
    #         tmp2.append(validity.cs_index(data[i], membership[i], clustter[i]))
    #         tmp2.append(validity.f1_score(target[i], np.argmax(membership[i], axis=1)))
    #         tmp2.append(validity.accuracy_score(target[i], np.argmax(membership[i], axis=1)))
    #         tmp.loc[len(tmp)]=tmp2
    #     print(tmp)
    


def validity2( data: np.ndarray, clustter: np.ndarray, membership: np.ndarray, target: np.ndarray):
    tmp=pd.DataFrame(columns=["DB", "PC", "CE", "S", "CH", "SI", "FHV", "CS", "AC"])
    for i in range (3):
        tmp2=[]
        tmp2.append(validity.davies_bouldin(data[i], np.argmax(membership[i], axis=1)))
        tmp2.append(validity.partition_coefficient(membership[i]))
        tmp2.append(validity.classification_entropy(membership[i]))
        tmp2.append(validity.separation(data[i], membership[i], clustter[i]))
        tmp2.append(validity.calinski_harabasz(data[i], np.argmax(membership[i], axis=1)))
        tmp2.append(validity.silhouette(data[i], np.argmax(membership[i], axis=1)))
        tmp2.append(validity.hypervolume(membership[i]))
        tmp2.append(validity.cs(data[i], membership[i], clustter[i]))
        tmp2.append(validity.accuracy_score(target[i], np.argmax(membership[i], axis=1)))
        tmp.loc[len(tmp)]=tmp2
    print(tmp)

if __name__=="__main__":

    np.random.seed(42)


    imgpr=image_pr(['b1_1024x1024.tif', 'b2_1024x1024.tif', 'b3_1024x1024.tif', 'b4_1024x1024.tif'])
    color=np.array([[0,0,255,255],[128, 128, 128,255],[0,255,0,255],[1,192,255,255],[0,128,0,255],[0,64,0,255]])
    data=imgpr.read_image().reshape(-1, 4)
    none_black=np.all(data!=[0,0,0,0], axis=1)
    index=np.where(none_black==True)[0].tolist()
    cfcm1=cfcm(20, 1e-4, 1, 2, data[index], 3, [6,6,6])
    cfcm1.run()

    imgpr.process(cfcm1.list_u, cfcm1.list_v, 3, 'hanoi_s2cfc_check.tif', index=index, color=color)


    # tmp=pd.read_excel("Dry_Bean_Dataset.xlsx")
    # data, target=np.array(tmp.iloc[:, 0:16]), pd.factorize(np.array(tmp.iloc[:,16]))[0]
    # tmp=np.random.choice(np.arange(0,len(data)), size=[len(data)],replace=False)
    # data=data[tmp]
    # target=target[tmp]


    # tmp=load_iris()
    # data, target = tmp['data'], tmp['target']

    
    # cfcm1=cfcm(100, 1e-4, 0, 2, data, 3, [3,3,3], target)
    # # print(cfcm1.run())
    # print('\nChi so CFCM\n')
    # validity2(cfcm1.list_data, cfcm1.list_v, cfcm1.list_u, cfcm1.list_target)
    # print('\n\n')