#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang

"""

import os
from sklearn import metrics
from scipy import interpolate
from keras.models import load_model
from utils.utils import *
from loss.loss import *
from network_models.network_module import *
import sys
from keras.utils import to_categorical
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description="Cross-spectral Patch Matching Demo")
    #parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    #parser.add_argument('--pretrained_weights', action='store_true', help='Use pretrained weights')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    #parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--dataset', type=str, default='VIS-NIR', choices=['VIS-NIR', 'OS', 'SEN1-2'], help='Dataset to use')
    parser.add_argument('--model', type=str, default='RRL', choices=['SCFDM', 'AFD', 'MFD', 'EFR', 'FIL', 'RRL'], help='Model architecture')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    #learning_rate = args.learning_rate
    #pretrained_weights = args.pretrained_weights
    batch_size = args.batch_size
    #epochs = args.epochs
    choose_dataset = args.dataset
    choose_model = args.model

    input_model = access_model(choose_model)
    out_dir_name = f"{choose_model}__{choose_dataset}"

    out_file_name = 'test_out_' + out_dir_name + '.txt'
    root_path = os.path.abspath('.')
    log_path = os.path.join(root_path, os.path.join('logs', out_dir_name))
    fig_path = os.path.join(root_path, os.path.join('figure', out_dir_name))
    model_path = os.path.join(root_path, os.path.join('model', out_dir_name))
    make_dir(log_path)
    make_dir(fig_path)
    make_dir(model_path)


    if choose_model == 'RRL':
        model_out_path = os.path.join(model_path, 'match_model_best_test.hdf5')
    else:
        model_out_path = os.path.join(model_path, 'match_model_best.hdf5')

    model = load_model(model_out_path,custom_objects={'InstanceNormalization':InstanceNormalization,'p_mse_loss_01':p_mse_loss_01,'large_margin_cosine_loss':large_margin_cosine_loss})
    testout_path = os.path.join(log_path, out_file_name)


    if choose_dataset == 'VIS-NIR':
        data_path = os.path.join(root_path, 'data','vis-nir')
        img_file_list = ['field.npy', 'forest.npy', 'indoor.npy', 'mountain.npy', 'oldbuilding.npy', 'street.npy', 'urban.npy',
                         'water.npy']
        label_file_list = ['field_label.npy', 'forest_label.npy', 'indoor_label.npy', 'mountain_label.npy',
                           'oldbuilding_label.npy', 'street_label.npy', 'urban_label.npy', 'water_label.npy']
    elif choose_dataset == 'OS':
        data_path = os.path.join(root_path, 'data', 'os')
        img_file_list = ['os_test_image.npy']
        label_file_list = ['os_test_label.npy']
    elif choose_dataset == 'SEN1-2':
        data_path = os.path.join(root_path, 'data', 'sen1-2')
        img_file_list = ['sne_test_image.npy']
        label_file_list = ['sne_test_label.npy']
    else:
        print("Error!!!! Please input right dataset name!!!")
        sys.exit(0)


    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    fpr95_list = []
    AUC_list = []

    for i in range(len(img_file_list)):
        error_0_list = []
        error_1_list = []
        count_0 = 0
        count_1 = 0
        print("加载数据中.........................................")
        img = np.load(os.path.join(data_path, img_file_list[i]))
        label = np.load(os.path.join(data_path, label_file_list[i]))
        label_multiclass = to_categorical(label, num_classes=2)
        print("---------------------------------------------------------")
        print(img.shape)
        print(label.shape)
        img = img / 255.0
        img_0 = img[:, 0].copy()
        img_1 = img[:, 1].copy()
        img_0 = np.expand_dims(img_0, axis=3)
        img_1 = np.expand_dims(img_1, axis=3)

        if choose_model == 'SCFDM':
            loss = model.evaluate([img_0, img_1], [label_multiclass, label], batch_size=batch_size, verbose=1)
            loss_out = loss[0]
            acc_out = loss[3]
        elif choose_model == 'AFD':
            loss = model.evaluate([img_0, img_1], [label_multiclass, label_multiclass], batch_size=batch_size, verbose=1)
            loss_out = loss[0]
            acc_out = loss[3]
        elif choose_model == 'MFD':
            loss = model.evaluate([img_0, img_1, img_0, img_1, img_0, img_1, img_0, img_1], [label,label,label,label,label], batch_size=batch_size, verbose=1)
            loss_out = loss[0]
            acc_out = loss[6]
        elif choose_model == 'EFR':
            loss = model.evaluate([img_0, img_1, img_0, img_1], [label, label, label], batch_size=batch_size, verbose=1)
            loss_out = loss[0]
            acc_out = loss[4]
        elif choose_model == 'FIL':
            loss = model.evaluate([img_0, img_1], [label, label, label, label, label, label], batch_size=batch_size, verbose=1)
            loss_out = loss[0]
            acc_out = loss[7]
        elif choose_model == 'RRL':
            loss = model.evaluate([img_0, img_1], [label], batch_size=batch_size, verbose=1)
            loss_out = loss[0]
            acc_out = loss[1]
        else:
            print("Error!!!! Please input right model name!!!")
            sys.exit(0)

        img_name = img_file_list[i].split('.')[0]
        print("{}测试的loss:{}".format(img_name, loss_out))
        print("{}测试的accuracy:{}".format(img_name, acc_out))

        with open(testout_path, "a+") as f:
            f.write(str(img_name) + "测试的loss:" + str(loss_out))
            f.write("\n")
            f.write(str(img_name) + "测试的accuracy:" + str(acc_out))
            f.write("\n")
            f.close()

        if choose_model == 'SCFDM' or choose_model == 'AFD':
            out_linshi = model.predict([img_0, img_1], batch_size=batch_size)
            out_linshi = out_linshi[0]
            out = out_linshi[:,1]
            out0_1 = np.argmax(out_linshi, axis=-1)
        elif choose_model == 'RRL':
            out = model.predict([img_0, img_1], batch_size=batch_size)
            #out = out[0]
            out0_1 = np.where(out > 0.5, 1, 0)
        else:
            out = model.predict([img_0, img_1], batch_size=batch_size)
            out = out[0]
            out0_1 = np.where(out > 0.5, 1, 0)


        img_num = len(img)
        for j in range(img_num):
            if (label[j] == out0_1[j]):
                continue
            else:
                if (label[j] == 1):
                    error_0_list.append(j)
                    count_0 = count_0 + 1
                else:
                    error_1_list.append(j)
                    count_1 = count_1 + 1
        save_root_path = os.path.join(fig_path, str(img_name))
        #save_error_image(img, error_0_list, error_1_list, save_root_path)
        with open(testout_path, "a+") as f:
            #f.write(str(img_name) + "误判为0的图像序号:" + str(error_0_list))
            f.write("\n")
            f.write(str(img_name) + "误判为0的个数:" + str(count_0))
            f.write("\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")
            #f.write(str(img_name) + "误判为1的图像序号:" + str(error_1_list))
            f.write("\n")
            f.write(str(img_name) + "误判为1的个数:" + str(count_1))
            f.write("\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")
            f.close()


        # accuracy = metrics.accuracy_score(label, out0_1)
        # precision = metrics.precision_score(label, out0_1)
        # recall = metrics.recall_score(label, out0_1)
        # f1 = metrics.f1_score(label, out0_1)
        fpr, tpr, thresholds = metrics.roc_curve(label, out)
        fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
        # area = metrics.roc_auc_score(label, out)

        # accuracy_list.append(accuracy)
        # precision_list.append(precision)
        # recall_list.append(recall)
        # f1_list.append(f1)
        fpr95_list.append(fpr95)
        # AUC_list.append(area)
        # print("{}测试的准确率输出:{}".format(img_name, accuracy))
        # print("{}测试的精准率输出:{}".format(img_name, precision))
        # print("{}测试的召回率输出:{}".format(img_name, recall))
        # print("{}测试的F1值输出:{}".format(img_name, f1))
        print("{}测试的fpr95输出:{}".format(img_name, fpr95))
        # print("{}测试的AUC输出:{}".format(img_name, area))

        with open(testout_path, "a+") as f:
            # f.write(str(img_name) + "测试的准确率:" + str(accuracy))
            # f.write("\n")
            # f.write(str(img_name) + "测试的精准率:" + str(precision))
            # f.write("\n")
            # f.write(str(img_name) + "测试的召回率:" + str(recall))
            # f.write("\n")
            # f.write(str(img_name) + "测试的F1值:" + str(f1))
            # f.write("\n")
            f.write(str(img_name) + "测试的fpr95:" + str(fpr95))
            f.write("\n")
            # f.write(str(img_name) + "测试的AUC:" + str(area))
            # f.write("\n")
            f.write("\n")
            f.write("\n")
            f.close()

        # plt.figure()
        # plt.plot(fpr, tpr, label='ROC curve (area=%0.2f)' % area)
        # plt.legend()
        # figout_name = str(img_name) + '_ROC.png'
        # plt.savefig(os.path.join(fig_path, figout_name))

    # accuracy_mean = np.mean(accuracy_list)
    # precision_mean = np.mean(precision_list)
    # recall_mean = np.mean(recall_list)
    # f1_mean = np.mean(f1_list)
    fpr95_mean = np.mean(fpr95_list)
    # AUC_mean = np.mean(AUC_list)

    # print("所有测试集的Accuracy_mean输出:", accuracy_mean)
    # print("所有测试集的Precision_mean输出:", precision_mean)
    # print("所有测试集的Recall_mean输出:", recall_mean)
    # print("所有测试集的F1_mean输出:", f1_mean)
    print("所有测试集的Fpr95_mean输出:", fpr95_mean)
    # print("所有测试集的AUC_mean输出:", AUC_mean)
    print("---------------------------------------------------------")
    with open(testout_path, "a+") as f:
        # f.write("所有测试集的accuracy_mean输出:" + str(accuracy_mean))
        # f.write("\n")
        # f.write("所有测试集的precision_mean输出:" + str(precision_mean))
        # f.write("\n")
        # f.write("所有测试集的recall_mean输出:" + str(recall_mean))
        # f.write("\n")
        # f.write("所有测试集的f1_mean输出:" + str(f1_mean))
        # f.write("\n")
        f.write("所有测试集的fpr95_mean输出:" + str(fpr95_mean))
        f.write("\n")
        # f.write("所有测试集的AUC_mean输出:" + str(AUC_mean))
        # f.write("\n")
        f.close()
    print("Done!!!!!!!!!!!!!!!!!!")