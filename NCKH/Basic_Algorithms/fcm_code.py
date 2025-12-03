import numpy as np
import pandas as pd
from ..Validity import validity
from ..Process.image_process import image_pr

from sklearn.datasets import load_iris
# from Process.image_process import image_pr

class fcm():
    def __init__(self, max_iter, eps, m):
        self.max_iter=max_iter
        self.eps=eps
        self.m=m

    def update_v(self, u: np.ndarray, data: np.ndarray):

        u_pw_m=u.T ** self.m
        tu=np.dot(u_pw_m, data)
        mau=np.sum(u_pw_m, axis=1, keepdims=True)
        return tu/mau


    def update_u(self, data: np.ndarray, v: np.ndarray):

        tu=np.linalg.norm(data[:, np.newaxis, :]-v, axis=2)
        mau=tu.T[:, : , np.newaxis]
        return 1/np.sum((tu/mau) ** (2/(self.m-1)), axis=0)



    def start(self, data: np.ndarray, num_of_clus: int, init_centroid=None):

        u=np.random.rand(len(data), num_of_clus)
        u=u/np.sum(u, axis=1, keepdims=True)
        if init_centroid is not None:
            v=init_centroid
        else:
            v=self.update_v(u, data)
        
        
        for i in range(self.max_iter):
            old_u=u.copy()
            u=self.update_u(data, v)
            v=self.update_v(u, data)

            tmp2=np.linalg.norm(u-old_u, ord=2)
            if tmp2<self.eps:
                self.u=u
                self.v=v 
                return u, v, i

        return u,v,i
    
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
# def validity2( data: np.ndarray, clustter: np.ndarray, membership: np.ndarray, target: np.ndairay):
#     tmp=pd.DataFrame(columns=["DB", "PC", "CE", "S", "CH", "SI", "FHV", "CS", "AC"])
#     for i in range (3):
#         tmp2=[]
#         tmp2.append(validity.davies_bouldin(data[i], np.argmax(membership[i], axis=1)))
#         tmp2.append(validity.partition_coefficient(membership[i]))
#         tmp2.append(validity.classification_entropy(membership[i]))
#         tmp2.append(validity.separation(data[i], membership[i], clustter[i]))
#         tmp2.append(validity.calinski_harabasz(data[i], np.argmax(membership[i], axis=1)))
#         tmp2.append(validity.silhouette(data[i], np.argmax(membership[i], axis=1)))
#         tmp2.append(validity.hypervolume(membership[i]))
#         tmp2.append(validity.cs(data[i], membership[i], clustter[i]))
#         tmp2.append(validity.accuracy_score(target[i], np.argmax(membership[i], axis=1)))
#         tmp.loc[len(tmp)]=tmp2
#     print(tmp)


def run():
    np.random.seed(42)
    
    imgpr=image_pr(['NCKH/HaNoi2-30/b2_30_3711x3046.tif', 'NCKH/HaNoi2-30/b3_30_3711x3046.tif', 'NCKH/HaNoi2-30/b4_30_3711x3046.tif', 'NCKH/HaNoi2-30/b5_30_3711x3046.tif'])
    color=np.array([[0, 128, 0, 255],[128, 128, 128, 255],[0, 255, 0, 255],[1, 192, 255, 255],[0, 0, 255, 255],[0, 64, 0, 255]])
    data=imgpr.read_image()
    none_black=np.all(data!=[0,0,0,0], axis=1)
    index=np.where(none_black==True)[0].tolist()

    # fcmrun=fcm(1000, 1e-3, 2)
    # u, v, i=fcmrun.start(data=data[index], num_of_clus=6)
    fcm_run=fcm(1000, 1e-4, 2)
    u, v, i=fcm_run.start(data=data[index], num_of_clus=6)

    imgpr.process([u], [v], 1, 'fcm_check.tif', index=index, color=color)


if __name__=='__main__':
    np.random.seed(42)
    
    imgpr=image_pr(['HaNoi2-30/b2_30_3711x3046.tif', 'HaNoi2-30/b3_30_3711x3046.tif', 'HaNoi2-30/b4_30_3711x3046.tif', 'HaNoi2-30/b5_30_3711x3046.tif'])
    color=np.array([[0, 128, 0, 255],[128, 128, 128, 255],[0, 255, 0, 255],[1, 192, 255, 255],[0, 0, 255, 255],[0, 64, 0, 255]])
    data=imgpr.read_image()
    none_black=np.all(data!=[0,0,0,0], axis=1)
    index=np.where(none_black==True)[0].tolist()

    # fcmrun=fcm(1000, 1e-3, 2)
    # u, v, i=fcmrun.start(data=data[index], num_of_clus=6)
    fcm_run=fcm(1000, 1e-4, 2)
    u, v, i=fcm_run.start(data=data[index], num_of_clus=6)

    imgpr.process([u], [v], 1, 'fcm_check.tif', index=index, color=color)

    # tmp=pd.read_excel("Dry_Bean_Dataset.xlsx")
    # data, target=np.array(tmp.iloc[:, 0:16]), pd.factorize(np.array(tmp.iloc[:,16]))[0]
    # rand_index=np.random.choice(np.arange(len(data)), size=len(data), replace=False)
    # data, target=data[rand_index], target[rand_index]
    # fcm1=fcm(1000, 1e-4, 2)
    # u, v, i=fcm1.start(data=data, num_of_clus=3)
    # validity2(data, v, u, target=target)

    # data=load_iris()
    # x=data['data']
    # y=data['target']
    # fcm1=fcm(1000, 1e-4, 2)
    # u, v, i=fcm1.start(data=x, num_of_clus=3)
    # print('\nChi so FCM\n')
    # validity2(x, v, u, target=y)
    # print('\n\n')
