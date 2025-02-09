#coding=gbk
'''
@author: yuchuang
'''
import os 
import numpy as np 
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import cv2
import importlib


def make_dir(path):
    if os.path.exists(path)==False:
        os.makedirs(path)
        
        

def save_fig(train_loss_list,path,label):
 
    plt.figure()
    plt.plot(train_loss_list, 'r')
    plt.legend([label],loc='best')
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.savefig(path)
    

def save_fig1(train_loss_list,train_acc_list,val_loss_list,val_acc_list,path):
 
    plt.figure()
    plt.plot(train_loss_list, 'r')
    plt.plot(train_acc_list, 'b')
    plt.plot(val_loss_list, 'g')
    plt.plot(val_acc_list, 'y')
    plt.legend(["train_loss","train_acc","val_loss","val_acc"],loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Index')
    plt.savefig(os.path.join(path,'index_out.png'))
    

def save_loss(train_loss_list,val_loss_list,path):
    plt.figure()
    plt.plot(train_loss_list, 'r')
    plt.plot(val_loss_list, 'g')
    plt.legend(["train_loss","val_loss"],loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(path,'Loss.png'))
    

def save_acc(train_acc_list,val_acc_list,path):
    plt.figure()
    plt.plot(train_acc_list, 'r')
    plt.plot(val_acc_list, 'g')
    plt.legend(["train_acc","val_acc"],loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.savefig(os.path.join(path,'Acc.png'))
    

def img_standard(img):
    img_mean = np.mean(img)
    img_std = np.std(img)
    img_new = (img - img_mean)/img_std
    return img_new


def img_standard_muli_dim(img):
    img_new = img.copy()
    for i in range(len(img)):
        img_mean = np.mean(img[i])
        img_std = np.std(img[i])
        img_new[i] = (img[i] - img_mean)/img_std
    return img_new


def img_normal(img):
    img_min = np.min(img)
    img_max = np.max(img)
    img_section = img_max - img_min
    img_new = (img - img_min)/img_section
    return img_new


def img_normal_muli_dim(img):
    img_new = img.copy()
    for i in range(len(img)):
        img_min = np.min(img[i])
        img_max = np.max(img[i])
        img_section = img_max - img_min
        img_new[i] = (img[i] - img_min)/img_section
    return img_new


def gen4(rgb_data,nir_data,label,batch_size):
    leng = len(label)
    rgb_batch = np.zeros((batch_size,64,64,1))
    nir_batch = np.zeros((batch_size,64,64,1))
    label_batch = np.zeros(batch_size)
    
    temp_list = []
    for i in range(batch_size):
        temp = np.random.randint(0,leng)
        while(temp in temp_list):
            temp = np.random.randint(0,leng)
        temp_list.append(temp)
        rgb_batch[i] = rgb_data[temp]   
        nir_batch[i] = nir_data[temp]
        label_batch[i] = label[temp]
        
    rgb_batch = np.array(rgb_batch,dtype='uint8')
    nir_batch = np.array(nir_batch,dtype='uint8')
    
    seq1 =  iaa.SomeOf((0,2),[
        
        iaa.GaussianBlur(0,1),
        iaa.GammaContrast((0.5, 1.5)),
        iaa.ScaleX((0.95,1.05),mode="symmetric"),
        iaa.ScaleY((0.95,1.05),mode="symmetric"),
        iaa.TranslateX(percent=(-0.05,0.05),mode="symmetric"),
        iaa.TranslateY(percent=(-0.05,0.05),mode="symmetric"),
        #iaa.PiecewiseAffine(scale=(0,0.05),mode="symmetric"),
        iaa.AdditiveGaussianNoise(scale=0.02*255)
        ],random_order=True)# 随机顺序作用在图片上
    
    
    seq2 = iaa.SomeOf((0,2),[
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270)],
        random_order=True)# 随机顺序作用在图片上
    
    nir_aug = seq1(images=nir_batch) 
    rgb_aug = seq1(images=rgb_batch)
    rgb_aug2, nir_aug2 = seq2(images=rgb_aug, segmentation_maps=nir_aug)
    
    
    return rgb_aug2,nir_aug2,label_batch



def no_gen4(rgb_data,nir_data,label,batch_size):
    leng = len(label)
    rgb_batch = np.zeros((batch_size,64,64,1))
    nir_batch = np.zeros((batch_size,64,64,1))
    label_batch = np.zeros(batch_size)
    
    temp_list = []
    for i in range(batch_size):
        temp = np.random.randint(0,leng)
        while(temp in temp_list):
            temp = np.random.randint(0,leng)
        temp_list.append(temp)
        rgb_batch[i] = rgb_data[temp]   
        nir_batch[i] = nir_data[temp]
        label_batch[i] = label[temp]
        
    rgb_batch = np.array(rgb_batch,dtype='uint8')
    nir_batch = np.array(nir_batch,dtype='uint8')
    return rgb_batch,nir_batch,label_batch


def make_filedir(root_path,out_dir_name,time1):
    log_path = os.path.join(root_path,os.path.join('logs',out_dir_name))
    make_dir(log_path)
    
    
    new_name1 = "history"+time1+".txt"
    history_path = os.path.join(log_path,new_name1)
    
    new_name4 = "testout"+time1+".txt"
    testout_path= os.path.join(log_path,new_name4)
    
    
    fig_path = os.path.join(root_path,os.path.join('figure',out_dir_name))
    make_dir(fig_path)
    
    
    model_path = os.path.join(root_path,os.path.join('model',out_dir_name))
    make_dir(model_path)
    return log_path,history_path,testout_path,fig_path,model_path


def save_error_image(img,error_0_index,error_1_index,save_root_path):
    make_dir(os.path.join(save_root_path,"error_to_0"))
    make_dir(os.path.join(save_root_path,"error_to_1"))
    error_to_0_path = os.path.join(save_root_path,"error_to_0")
    error_to_1_path = os.path.join(save_root_path,"error_to_1")
    img = img*255
    for i in error_0_index:
        new_name = str(i)+'.png'
        img_out = np.zeros((64,130))
        error_index = i
        img1 = img[error_index,0]
        img2 = img[error_index,1]
        img_out[:,0:64] = img1
        img_out[:,66:130]=img2
        cv2.imwrite(os.path.join(error_to_0_path,new_name),img_out)
    for j in error_1_index:
        new_name = str(j)+'.png'
        img_out = np.zeros((64,130))
        error_index = j
        img1 = img[error_index,0]
        img2 = img[error_index,1]
        img_out[:,0:64] = img1
        img_out[:,66:130]=img2
        cv2.imwrite(os.path.join(error_to_1_path,new_name),img_out)
    print("误判为0和误判为1的图像保存完成！！！！！！！")




def access_model(choose_model):
    module_name = f"network_models.{choose_model}"
    module = importlib.import_module(module_name)
    model_func = getattr(module, choose_model)
    return model_func

