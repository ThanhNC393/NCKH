import numpy as np
import pandas as pd
from .cfcm_code import cfcm
from sklearn.datasets import load_iris, load_wine
from ..Process.image_process import image_pr

class sscfcm(cfcm):
    def __init__(self, max_iter: int, espsilon: float, beta: int, m: int, data: np.ndarray, num_of_data_site: int, num_of_clus: list, per_tar: int, target=None ) -> None:
        super().__init__(max_iter, espsilon, beta, m, data, num_of_data_site, num_of_clus, target)
        self.list_u_bar=[]
        self.list_u=list(map(lambda a: a.T, self.list_u))
        self.l=[]
        np.random.seed(42)
        #Tạo u bar
        for i in range(self.num_of_data_site): 
            tmp=np.random.choice(np.arange(0, len(self.list_data[i])), size=(len(self.list_data[i])//100)*per_tar, replace=False)
            tmp2=np.zeros([self.num_of_clus[i], len(self.list_data[i])])
            tmp2[self.list_target[i][tmp], tmp]=1
            self.l.append(tmp)
            self.list_u_bar.append(tmp2)

    #Cập nhật u
    def update_u(self, list_v_nga: list, i: int):
        list_v_nga=np.array(list_v_nga)
        tu1=self.list_data[i]-np.reshape(self.list_v[i], newshape=[len(self.list_v[i]), 1, -1])
        tu2=self.list_v[i]-list_v_nga
        tu=np.sum(np.power(tu1, 2), axis=2)+self.beta*np.sum(np.power(tu2, 2), axis=1, keepdims=True)
        mau=np.power((1/tu), 1/(self.m-1))
        mau=np.sum(mau, axis=0, keepdims=True)
        self.list_u[i]=(np.power(tu, -1))/mau

    #Cập nhật v
    def update_v(self, v_nga: list, i:int):
        x, y=self.list_u[i].shape
        a, b=v_nga.shape
        u_power=np.power(self.list_u[i]-self.list_u_bar[i], self.m)
        cp1=np.sum(u_power.reshape(x, y, 1)*self.list_data[i], axis=1)
        cp2=np.sum(v_nga.reshape(a, b, 1)*u_power.reshape(x, 1, y), axis=2)*self.beta
        cp3=np.sum(u_power,axis=1, keepdims=True)*(self.beta+1)
        self.list_v[i]=(cp1+cp2)/(cp3)
        
if __name__=="__main__":

    # tmp=pd.read_excel("Dry_Bean_Dataset.xlsx")
    # data, target=np.array(tmp.iloc[:, 0:16]), pd.factorize(np.array(tmp.iloc[:,16]))[0]
    # tmp=np.random.choice(np.arange(0,len(data)), size=[len(data)],replace=False)
    # data, target=data[tmp], target[tmp]
    # sscfcm1=sscfcm(1, 1e-4, 0.5, 2, data, 3, [7,7,7],per_tar=30, target=target) 
    # sscfcm1.run()
    # print(sscfcm1.list_v[0].shape)
    # list_u=list(map(lambda a: a.T, sscfcm1.list_u))
    # sscfcm1.validity2(sscfcm1.list_data, sscfcm1.list_v, list_u, sscfcm1.list_target)



    imgpr=image_pr(['b1_1024x1024.tif', 'b2_1024x1024.tif', 'b3_1024x1024.tif', 'b4_1024x1024.tif'])
    color=np.array([[0, 0, 255, 255],[128, 128, 128, 255],[0, 255, 0, 255],[1, 192, 255, 255],[0, 128, 0, 255],[0, 64, 0, 255]])
    data=imgpr.read_image().reshape(-1, 4)
    none_black=np.all(data!=[0,0,0,0], axis=1)
    index=np.where(none_black==True)[0].tolist()

    sscfcm1=sscfcm(max_iter=50, espsilon=1e-4, beta=1, m=2, data=data[index], num_of_data_site=3, num_of_clus=[6,6,6], per_tar=30)
    sscfcm1.run()
    list_u=list(map(lambda a: a.T, sscfcm1.list_u))
    imgpr.process(list_u, sscfcm1.list_v, sscfcm1.num_of_data_site, data, imgpr.x, imgpr.y, imgpr.z, 'aaaaaa.tif', 2, index, color)
    pass