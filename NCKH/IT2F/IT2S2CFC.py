import numpy as np
from sklearn.datasets import load_iris, load_wine
from ..Basic_Algorithms.ssfcm_code import ssfcm
from ..Validity import validity
import pandas as pd
from ..Process.image_process import image_pr
from ..Basic_Algorithms.fcm_code import fcm
from ..Basic_Algorithms.s2cfc_code import s2cfcm, ln, exp, validity2
from .IT2FCM_Thanh import it2fcm, num, sortt




class it2s2cfc:
    def __init__(self, data: np.ndarray, label: np.ndarray, gamma_1: float, gamma_2: float, 
                 num_of_site: int, num_of_clus: int, per_tar: int = 30, 
                 theta: int = 1, max_iter: int = 1000, esp: float = 1e-4):
        self.data_goc, self.label_goc, self.gamma_1, self.gamma_2 = data, label, gamma_1, gamma_2
        self.num_of_site, self.per_tar, self.theta = num_of_site, per_tar, theta
        self.max_iter, self.esp, self.num_of_clus = max_iter, esp, num_of_clus
        self.u_bar, self.u, self.v, self.l=[], [], [], []


    def split_data(self):
        list_order=[[] for i in range(self.num_of_site)]
        list_label=np.unique(self.label_goc)
        label_spl=list(np.where(self.label_goc==i)[0] for i in list_label)
        self.data, self.label, self.order = [], [], []
        list_data=list([] for i in range(self.num_of_site))
        list_label=list([] for i in range(self.num_of_site))
        for i in range(len(label_spl)):
            tmp4=np.array_split(label_spl[i], self.num_of_site)
            for j in range(self.num_of_site):
                list_order[j].append(tmp4[j])
                list_data[j].append(self.data_goc[tmp4[j].tolist()])
                list_label[j].append(self.label_goc[tmp4[j].tolist()])
        for i in range(self.num_of_site):
            self.data.append(np.concatenate(list_data[i], axis=0))
            self.label.append(np.concatenate(list_label[i], axis=0))
            self.order.append(np.concatenate(list_order[i], axis=0))
        self.data2=self.data.copy()


    def run_ssfcm_local(self):
        for i in range(self.num_of_site):
            run_ssfcm=ssfcm(max_iter=self.max_iter, eps=self.esp, m=2)
            run_ssfcm.start(data=self.data[i], num_of_clus=self.num_of_clus, per=self.per_tar, target=self.label[i])
            self.u_bar.append(run_ssfcm.u_bar)

            self.u.append(run_ssfcm.u)
            self.v.append(run_ssfcm.v)
            self.l.append(run_ssfcm.l)


    def make_u_star(self):
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

    
    def gh_algorithm(self, arr1:np.ndarray, arr2:np.ndarray):
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
    

    def make_order(self):
        list_order=[[] for i in range(self.num_of_site)]
        for i in range(self.num_of_site):
            for j in range(len(self.u_star[i])):
                order=self.gh_algorithm(self.u[i].T, self.u_star[i][j].T)
                sorted_i=sorted(order)
                sorted_v=list(order[z] for z in sorted_i)
                list_order[i].append(sorted_v)
        return list_order


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
    

    def compute_u_1 (self, index, gamma, beta , distance, u_star_ordered):
        l=self.l[index][:, np.newaxis]
        b=beta[:, np.newaxis, np.newaxis]

        minn, maxx=np.min(distance), np.max(distance)
        distance=(distance-minn)/(maxx-minn)
        tu=  -gamma  -  np.power(distance, 2)  +  l*(ln(self.u_bar[index])-1)  +  np.sum(b*(1-l)*(ln(u_star_ordered)-1), axis=0)
        mau=  gamma  +  l  +  np.sum(b*(1-l), axis=0)
        ex=exp(tu/mau)
        return ex/np.sum(ex, axis=1, keepdims=True)


    def compute_u_2(self, index, beta, distance, u_star_ordered):
        # distance=np.linalg.norm(self.x[:, np.newaxis, :]-self.v, axis=2)
        u1=self.compute_u_1(index=index, gamma=self.gamma_1, beta=beta, 
                            distance=distance, u_star_ordered=u_star_ordered)
        u2=self.compute_u_1(index=index, gamma=self.gamma_2, beta=beta, 
                            distance=distance, u_star_ordered=u_star_ordered)
        # u1_u2=np.stack([u1, u2], axis=0)
        # moc_so_sanh=1/np.sum((distance.T[:, :, np.newaxis]/distance), axis=2).T
        # for i in range(len(self.x)):
        #     for j in range(self.number_of_centroid):
        #         if moc_so_sanh[i][j]<1/self.number_of_centroid:
        #             tmp=u1_u2[0][i][j]
        #             u1_u2[0][i][j]=u1_u2[1][i][j]
        #             u1_u2[1][i][j]=tmp
        # self.u=u1_u2
        u1_u2=np.stack([u1, u2], axis=0)
        index=np.where(u1_u2[0]>=u1_u2[1])
        tmp=u1_u2[0][index]
        u1_u2[0][index]=u1_u2[1][index]
        u1_u2[1][index]=tmp
        return u1_u2
    


    def compute_v(self, u1_u2, index):
        u_trung_binh=np.sum(u1_u2, axis=0)/2
        return np.sum(u_trung_binh.T[:, :, np.newaxis]*self.data[index],
                      axis=1)/np.sum(u_trung_binh.T, axis=1, keepdims=True)



    def sort_data(self):
        l2_index=[]
        for i in range(len(self.data)):
            data=self.data[i].T
            list_index=[]
            for j in range(len(data)):
                tmp=np.argsort(data[j])
                list_index.append(tmp)
                data[j]=data[j][tmp]
            self.data[i]=data.T
            l2_index.append(np.stack(list_index, axis=0))
        return l2_index
    

    
    def compute_c(self, lower_index, u_lower, upper_index, u_upper, index):
        c2=np.sum(self.data[index][lower_index]*u_lower[:, np.newaxis], axis=0) + np.sum(self.data[index][upper_index]*u_upper[:, np.newaxis], axis=0) 
        return c2/(np.sum(u_upper)+np.sum(u_lower))



    def karnik_algo(self, v, clus, mode, index, u1_u2):
        data=self.data[index].T
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
                lower_index=self.list_index[index][i][:list_k[i]+1]
                upper_index=self.list_index[index][i][list_k[i]+1:]
                if mode==1:
                    u_lower=u1_u2[:, lower_index, clus][0]
                    u_upper=u1_u2[:, upper_index, clus][1]
                else:
                    u_lower=u1_u2[:, lower_index, clus][1]
                    u_upper=u1_u2[:, upper_index, clus][0]
                v_new.append(self.compute_c(lower_index=lower_index, u_lower=u_lower, upper_index=upper_index, u_upper=u_upper, index=index))
                u_2.append(sortt(self.list_index[index][i],np.concatenate([u_lower, u_upper])))
            v_new=np.sum(np.array(v_new), axis=0)/len(list_k)
            if np.linalg.norm(v_new-v[clus]) < 1e-2:
                return v_new, np.sum(np.stack(u_2, axis=0).T, axis=1)/len(self.data[0][0])
            v[clus]=v_new
        return v_new, np.sum(np.stack(u_2, axis=0).T, axis=1)/len(self.data[0][0])
        



    def fit(self, num_of_pharse):
        self.split_data()

        self.run_ssfcm_local()
        self.u_star=self.make_u_star()
        self.list_order=self.make_order()
        self.list_index=self.sort_data()

        for i in range(num_of_pharse):
            if i!=0:
                self.u_star=self.make_u_star()
            for j in range(self.num_of_site):
                distance=np.linalg.norm(self.data[j][:, np.newaxis, :] - self.v[j], axis=2)
                u_star_orded=self.reorder(self.u_star[j].copy(), index=j)
                for z in range(self.max_iter):
                    b=self.caculate_beta(u_star=u_star_orded, index=j)
                    u1_u2=self.compute_u_2(index=j, beta=b, distance=distance, u_star_ordered=u_star_orded)
                    v_trung_binh=self.compute_v(u1_u2=u1_u2, index=j)

                    upper_centroid=[]
                    lower_centroid=[]
                    real_u=[]
                    for h in range(self.num_of_clus):
                        km_for_upper=self.karnik_algo(v=v_trung_binh, clus=h, mode=1, index=j, u1_u2=u1_u2)
                        upper_centroid.append(km_for_upper[0])

                        km_for_lower=self.karnik_algo(v=v_trung_binh, clus=h, mode=0, index=j, u1_u2=u1_u2)
                        lower_centroid.append(km_for_lower[0])

                        real_u.append((km_for_upper[1]+km_for_lower[1])/2)
                    upper_centroid=np.array(upper_centroid)
                    lower_centroid=np.array(lower_centroid)

                    hard_partioning=(upper_centroid+lower_centroid)/2
                    self.u[j]=np.stack(real_u, axis=0).T
                    if np.linalg.norm(self.v[j]-hard_partioning)<self.esp:
                        self.v[j]=hard_partioning
                        break
                    self.v[j]=hard_partioning

def run():
    data=load_iris()
    x=data['data']
    y=data['target']
    init_centroid=np.array([[4.8, 3, 1, 0.3],[4, 2, 3, 1],[6, 3, 6, 1]])


    it2s2cfc_object=it2s2cfc(data=x, gamma_1=1, gamma_2=2, num_of_site=3, 
                             per_tar=30, num_of_clus=3, label=y)

    it2s2cfc_object.fit(num_of_pharse=2)
    # from s2cfc_code import validity2
    validity2(data=it2s2cfc_object.data2, clustter=it2s2cfc_object.v, membership=it2s2cfc_object.u, target=it2s2cfc_object.label)
    

def run_image():
    np.random.seed(42)
    imgpr=image_pr(['NCKH/Data/b1_1024x1024.tif', 'NCKH/Data/b2_1024x1024.tif', 'NCKH/Data/b3_1024x1024.tif', 'NCKH/Data/b4_1024x1024.tif'])
    color=np.array([[0, 128, 0, 255],[128, 128, 128, 255],[0, 255, 0, 255],[1, 192, 255, 255],[0, 0, 255, 255],[0, 64, 0, 255]])
    data=imgpr.read_image()
    print('Tạo dữ liệu bán giám sát ...')
    fcmrun=fcm(1000, 1e-3, 2)
    print('Tạo thành công!\n\n')
    print('Bắt đàu phân cụm ...')
    it2s2cfc_object=it2s2cfc(data=data, label=np.argmax(list(fcmrun.start(data=data, num_of_clus=6))[0], axis=1), gamma_1=1, gamma_2=2, num_of_site=3, num_of_clus=6)
    it2s2cfc_object.fit(num_of_pharse=1)
    print('Phân cụm kết thúc!')
    imgpr.process(it2s2cfc_object.u, it2s2cfc_object.v, 3, 'hanoi_s2cfc_check.tif', color=color, order=it2s2cfc_object.order)



