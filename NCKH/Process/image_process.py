import rasterio as ra
import numpy as np
from PIL import Image
import tifffile as tiff
import matplotlib.pyplot as plt
# from ..IT2FCM.Border_IT2_FCM import sortt
# from ..IT2F.Border_IT2_FCM import sortt

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

class image_pr:

    #hàm khởi tạo đưa vào danh sách địa chỉ các ảnh viễn thám (ví dụ: imgpr=image_pr(['b1_1024x1024.tif', 'b2_1024x1024.tif', 'b3_1024x1024.tif', 'b4_1024x1024.tif']))
    def __init__(self, list_director:list):
        self.list_image=list_director
        self.number_image = len(self.list_image)



    def export(self, matrix, name):
        rgb_image_normalized = np.clip(matrix, 0, 255).astype('uint8')
        image = Image.fromarray(rgb_image_normalized)
        image.save(name)



    #sau đó gọi hàm này, nó sẽ trả về một danh sách dữ liệu là các điểm ảnh đã được duỗi ra (ví dụ: data=imgpr.read_image())
    #lấy data đi phâm cụm bình thường
    def read_image(self):
        tmp=[]
        for i in self.list_image:
            tmp.append((ra.open(i)).read().squeeze())
        image=np.clip(np.stack(tmp, axis=0), 0, 255).astype('uint8')
        image=image.transpose(1, 2, 0)
        image = Image.fromarray(image)
        result=np.array(image)
        self.x, self.y, self.z=result.shape
        self.data=result.reshape(-1,self.number_image)
        return self.data



    #sau khi thực hiện phân cụm, gọi hàm này để nó trả về ảnh đầu ra
    #list_u là danh sách các u của các data_site, nếu chỉ phân cụm một data_site thì để là [u]
    #list_v là danh sách các v của các data_site, nếu chỉ phân cụm một data_site thì để là [v]
    #num_of_data_site là số lượng các data_site, nếu chỉ phân cụm một data_site thì để là 1
    #name_output là tên của ảnh muốn xuất ra (ví dụ: 'anh_phan_cum.tif')
    #các tham số khác không cần để ý
    def process(self, list_u, list_v, num_of_data_site, name_output, mode=2, x=None, y=None, z=None, index=None, color=np.array([[0, 128, 0, 255],[128, 128, 128, 255],[0, 255, 0, 255],[1, 192, 255, 255],[0, 0, 255, 255],[0, 64, 0, 255]]), order=None):
        if mode==1:
            for i in range(num_of_data_site):
                list_u[i]=np.sum(list_v[i]*(list_u[i].reshape(-1, 9, 1)), axis=1)
            
        elif mode==2:
            list_u[0]=color[np.argmax(list_u[0], axis=1)]
            if(num_of_data_site>1):
                list2=list(map(lambda a: a.reshape(len(a), 1, -1), list_v[1:]))
                list2=np.stack(list2, axis=0)
                list2=list2-list_v[0]
                list2=np.linalg.norm(list2, axis=3)
                # list2=np.argmin(list2, axis=2)
                for i in range(0,num_of_data_site-1):
                    check3=[[j,z] for j in range(len(list2[i])) for z in range(len(list2[i][0]))]
                    check4=np.arange(len(check3)).reshape(*(list2[i].shape))
                    tmp=[]
                    tmp2=list2[i]
                    while len(tmp2)!=0:
                        check=[np.argmin(tmp2, axis=1), np.argmin(tmp2, axis=0)]
                        check1=dict(zip(range(len(check[0])),check[0]))
                        check2=dict(zip(range(len(check[1])),check[1]))
                        for t in check1:
                            if check2[check1[t]]==t:
                                tmp.append(check3[check4[t][check1[t]]])
                                tmp2=np.delete(tmp2, t, axis=0)
                                tmp2=np.delete(tmp2, check1[t], axis=1)
                                check4=np.delete(check4, t, axis=0)
                                check4=np.delete(check4, check1[t], axis=1)
                                break
                    list_u[i+1]=color[np.array(tmp)[:,1][np.argmax(list_u[i+1], axis=1)]]
        else: return
        if index is not None:
            self.data[index]=np.concatenate(list_u, axis=0)
        else: self.data=np.concatenate(list_u, axis=0)
        if order is not None:
            self.data=sortt(np.concatenate(order, axis=0).tolist(), self.data)
        if all([t is not None for t in [x,y,z]]):
            self.data=self.data.reshape(x, y, z)
        else:
            self.data=self.data.reshape(self.x, self.y, self.z)
        self.export(self.data, name_output)
        pass

if __name__=="__main__":
    tmp=image_pr(['b1_1024x1024.tif', 'b2_1024x1024.tif', 'b3_1024x1024.tif', 'b4_1024x1024.tif'])
    tmp2=tmp.read_image()
    
    # tmp.export(tmp.read_image(), "Nhap_output.tif")