import numpy as np
from sklearn.datasets import load_iris, load_wine
from .ssfcm_code import ssfcm
from ..Validity import validity
import pandas as pd
from ..Process.image_process import image_pr
from .fcm_code import fcm

def ln(arr1: np.array):
    arr2=arr1.copy()
    arr2[np.where(arr2==0)]=1e-300
    return np.log(arr2)

def exp(arr1: np.array):
    arr2=arr1.copy()
    arr2[np.where(arr2<-300)]=-np.inf
    return np.exp(arr2)
    
class s2cfcm:
    def __init__(
            self, data: np.ndarray,  gamma: int, theta: int, 
            max_iter: int, epsilon: float, num_of_site: int, 
            per_tar: int, num_of_clus: int, label:np.ndarray
    ):

        self.gamma, self.theta, self.max_iter = gamma, theta, max_iter
        self.per_tar, self.num_of_site, self.epsilon, self.num_of_clus = per_tar, num_of_site, epsilon, num_of_clus
        self.u_bar, self.u, self.v, self.l=[], [], [], []
        list_order=[[] for i in range(num_of_site)]

        list_label=np.unique(label)
        label_spl=list(np.where(label==i)[0] for i in list_label)
        self.data, self.label, self.order = [], [], []
        list_data=list([] for i in range(num_of_site))
        list_label=list([] for i in range(num_of_site))
        for i in range(len(label_spl)):
            tmp4=np.array_split(label_spl[i], num_of_site)
            for j in range(num_of_site):
                list_order[j].append(tmp4[j])
                list_data[j].append(data[tmp4[j].tolist()])
                list_label[j].append(label[tmp4[j].tolist()])
        for i in range(num_of_site):
            self.data.append(np.concatenate(list_data[i], axis=0))
            self.label.append(np.concatenate(list_label[i], axis=0))
            self.order.append(np.concatenate(list_order[i], axis=0))




    def run_ssfcm_local(self):
        for i in range(self.num_of_site):

            run_ssfcm=ssfcm(max_iter=1000, eps=1e-3, m=2)
            run_ssfcm.start(data=self.data[i], num_of_clus=self.num_of_clus, per=30, target=self.label[i])
            self.u_bar.append(run_ssfcm.u_bar)

            # run_ssfcm=gspfcm(a=1, b=1, eta=2)
            # run_ssfcm.train(data=self.data[i], label=self.label[i], num_of_clus=self.num_of_clus, ratio_label=self.per_tar)
            # self.u_bar.append(run_ssfcm.u_star)

            self.u.append(run_ssfcm.u)
            self.v.append(run_ssfcm.v)
            self.l.append(run_ssfcm.l)


    def u_star(self):
        list_u_star=[]
        
        for i in range(self.num_of_site):
            list_tmp=[]
            for j in range(self.num_of_site):
                if j!=i:
                    dis_xv=np.linalg.norm(self.data[i][:, np.newaxis, :]-self.v[j], axis=2)
                    dis_xv=dis_xv[:, :, np.newaxis] / dis_xv[:, np.newaxis, :]
                    dis_xv=np.sum(dis_xv ** 2, axis=2)
                    list_tmp.append(dis_xv**(-1))
            list_u_star.append(np.stack (list_tmp, axis=0))

        return list_u_star
    


    def gh_algorithm(self, arr1, arr2):
            x=arr1.shape[0]
            ar1=arr1[:, np.newaxis, np.newaxis, :].repeat(x, axis=2)
            ar2=arr2[np.newaxis, np.newaxis, :, :].repeat(x, axis=0)
            distance=np.concatenate([ar1, ar2], axis=1, dtype="float32")
            distance=1-np.sum(np.min(distance, axis=1), axis=2)/np.sum(np.max(distance, axis=1), axis=2)
            position_r=list(range(len(distance)))
            position_c=list(range(len(distance[0])))
            result={}
            while distance.size>0:
                min_r=np.argmin(distance, axis=1)
                min_c=np.argmin(distance, axis=0)
                for i in range(len(min_r)):
                    if min_c[min_r[i]]==i:
                        distance=np.delete(distance, min_r[i], axis=0)
                        distance=np.delete(distance, i, axis=1)
                        result[position_r[i]]=position_c[min_r[i]]
                        del position_r[i]
                        del position_c[min_r[i]]
                        break
            return result


    def make_order(self, u_star):
        self.list_order=[[] for i in range(self.num_of_site)]
        for i in range(self.num_of_site):
            for j in range(len(u_star[i])):
                order=self.gh_algorithm(self.u[i].T, u_star[i][j].T)
                sorted_i=sorted(order)
                sorted_v=list(order[z] for z in sorted_i)
                self.list_order[i].append(sorted_v)
        

    def reorder(self, u_star, index):
        u_star2=np.transpose(u_star, [0, 2, 1])
        for i in range(len(u_star2)):
            u_star2[i]=u_star2[i][self.list_order[index][i]]
        return np.transpose(u_star2, [0, 2, 1])


    def caculate_beta(self, u_star: list, index: int):

        l=self.l[index][:, np.newaxis]
        # nuy=np.sum(l*self.u_bar[index]*ln(self.u_bar[index]/u_star)+(1-l)*self.u[index]*ln(self.u[index]/u_star), axis=1)
        nuy=np.sum((1-l)*self.u[index]*ln(self.u[index]/u_star), axis=1)
        nuy=np.sum(nuy, axis=1)

        minn, maxx=np.min(nuy), np.max(nuy)
        nuy=(nuy-minn)/(maxx-minn)
        nuy=np.exp(-self.theta*nuy-1)
        return nuy/np.sum(nuy)


    def update_u(self, beta, u_star, index, distance:np.ndarray):

        l=self.l[index][:, np.newaxis]
        b=beta[:, np.newaxis, np.newaxis]

        minn, maxx=np.min(distance), np.max(distance)
        distance=(distance-minn)/(maxx-minn)
        tu=  -self.gamma  -  np.power(distance, 2)  +  l*(ln(self.u_bar[index])-1)  +  np.sum(b*(1-l)*(ln(u_star)-1), axis=0)
        mau=  self.gamma  +  l  +  np.sum(b*(1-l), axis=0)
        ex=exp(tu/mau)
        return ex/np.sum(ex, axis=1, keepdims=True)
        

    def update_v(self, index: int):
        tu=np.dot(self.u[index].T, self.data[index])
        mau=np.sum(self.u[index].T, axis=1, keepdims=True)
        return tu/mau
    

    def check_ter(self, u_old: np.ndarray, u_new: np.ndarray, mode: int):

        if mode==1:
            tmp=np.linalg.norm(u_new-u_old)
            if tmp<self.epsilon: 
                return True
            return False
        
        if mode==2:
            sum=0
            for i in range(len(u_old)):
                sum+=np.linalg.norm(u_old[i]-u_new[i])
            if sum<self.epsilon: return True
            return False
        

    def fit(self, number_of_pharse):
        self.run_ssfcm_local()
        u_star=self.u_star()
        self.make_order(u_star)
        # for i in range(self.max_iter):
        for i in range(number_of_pharse):
            if i!=0:
                u_star=self.u_star()
            for j in range(self.num_of_site):
                distance=np.linalg.norm(self.data[j][:, np.newaxis, :]-self.v[j], axis=2)
                u_cu_total=self.u.copy()
                u_star_reo=self.reorder(u_star[j].copy(), index=j) #u_star[j].copy()
                for z in range(self.max_iter):
                    b=self.caculate_beta(u_star_reo.copy(), j)
                    u_cu=self.u[j].copy()
                    self.u[j]=self.update_u(b, u_star_reo.copy(), j, distance)
                    self.v[j]=self.update_v(index=j)
                    if self.check_ter(u_old=u_cu, u_new=self.u[j], mode=1):
                        break
            if self.check_ter(u_old=u_cu_total, u_new=self.u, mode=2):
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
                    

# def validity1(data: list, clustter: list, membership: list, target: list):
#     tmp=pd.DataFrame(columns=["DI", "DB", "PC", "CE", "CH", "SI", "FHV", "CS", "S", "AC"])
#     for i in range(len(list(data))):
#         tmp2=[]
#         tmp2.append(validity.dunn(data[i], np.argmax(membership[i], axis=1)))
#         tmp2.append(validity.davies_bouldin(data[i], np.argmax(membership[i], axis=1)))
#         tmp2.append(validity.partition_coefficient(membership[i]))
#         tmp2.append(validity.classification_entropy(membership[i]))
#         tmp2.append(validity.calinski_harabasz(data[i], np.argmax(membership[i], axis=1)))
#         tmp2.append(validity.silhouette(data[i], np.argmax(membership[i], axis=1)))
#         tmp2.append(validity.hypervolume(membership[i]))
#         tmp2.append(validity.cs(data[i], membership[i], clustter[i]))
#         tmp2.append(validity.separation(data[i], membership[i], clustter[i]))
#         tmp2.append(validity.accuracy_score(target[i], np.argmax(membership[i], axis=1)))
#         tmp.loc[len(tmp)]=tmp2
#     print(tmp)            
    

def run():
    np.random.seed(42)
    imgpr=image_pr(['NCKH/Data/b1_1024x1024.tif', 'NCKH/Data/b2_1024x1024.tif', 'NCKH/Data/b3_1024x1024.tif', 'NCKH/Data/b4_1024x1024.tif'])
    color=np.array([[0, 128, 0, 255],[128, 128, 128, 255],[0, 255, 0, 255],[1, 192, 255, 255],[0, 0, 255, 255],[0, 64, 0, 255]])
    data=imgpr.read_image()
    none_black=np.all(data!=[0,0,0,0], axis=1)
    index=np.where(none_black==True)[0].tolist()
    print('Tạo dữ liệu bán giám sát ...')
    fcmrun=fcm(1000, 1e-3, 2)
    print('Tạo thành công!\n\n')
    print('Bắt đàu phân cụm ...')
    sscfcm_vip1=s2cfcm(data=data[index], gamma=1, theta=1, max_iter=50, epsilon=1e-3, num_of_site=3, per_tar=5, num_of_clus=6, label=np.argmax(list(fcmrun.start(data=data[index], num_of_clus=6))[0], axis=1))
    i=sscfcm_vip1.start(number_of_pharse=1)
    print('Phân cụm kết thúc!')
    imgpr.process(sscfcm_vip1.u, sscfcm_vip1.v, 3, 'hanoi_s2cfc_check.tif', index=index, color=color, order=sscfcm_vip1.order)


if __name__=='__main__':

    np.random.seed(42)
    imgpr=image_pr(['b1_1024x1024.tif', 'b2_1024x1024.tif', 'b3_1024x1024.tif', 'b4_1024x1024.tif'])
    color=np.array([[0, 128, 0, 255],[128, 128, 128, 255],[0, 255, 0, 255],[1, 192, 255, 255],[0, 0, 255, 255],[0, 64, 0, 255]])
    data=imgpr.read_image()
    none_black=np.all(data!=[0,0,0,0], axis=1)
    index=np.where(none_black==True)[0].tolist()
    print('Tạo dữ liệu bán giám sát ...')
    fcmrun=fcm(1000, 1e-3, 2)
    print('Tạo thành công!\n\n')
    print('Bắt đàu phân cụm ...')
    sscfcm_vip1=s2cfcm(data=data[index], gamma=1, theta=1, max_iter=50, epsilon=1e-3, num_of_site=3, per_tar=5, num_of_clus=6, label=np.argmax(list(fcmrun.start(data=data[index], num_of_clus=6))[0], axis=1))
    i=sscfcm_vip1.start(number_of_pharse=1)
    print('Phân cụm kết thúc!')
    imgpr.process(sscfcm_vip1.u, sscfcm_vip1.v, 3, 'hanoi_s2cfc_check.tif', index=index, color=color, order=sscfcm_vip1.order)





    # imgpr.process([u], [v], 1, imgpr.x, imgpr.y, imgpr.z, 'hanoi_s2cfc.tif', 2, index, color)
    # tmp=pd.read_excel("Dry_Bean_Dataset.xlsx")
    # data, target=np.array(tmp.iloc[:, 0:16]), pd.factorize(np.array(tmp.iloc[:,16]))[0]

    # tmp=load_iris()
    # data, target=tmp['data'], tmp['target']
    # np.random.seed(42)
    
    # tmp=sscfcm_vip(data=data, label=target, gamma=1e-1, theta=5e-1, max_iter=1000, epsilon=1e-5, num_of_site=3, per_tar=20, num_of_clus=3)
    # # print('\nGSPFCM x S2CFC\n')
    # print('Số lần lặp: {}\n'.format( tmp.start()))
    # print('\nChi so SSCFCM\n')
    # validity2(data=tmp.data, clustter=tmp.v, membership=tmp.u, target=tmp.label)
    # print('\n\n')


