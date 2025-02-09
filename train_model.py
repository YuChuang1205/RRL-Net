#!/usr/bin/python3
# coding=gbk
"""
@author: yuchuang
@time:
@desc:
"""

import sys
import time
import keras
from sklearn import metrics
from scipy import interpolate
from keras.models import *
from keras.layers import *
from utils.utils import *
from loss.loss import *
from keras.utils import to_categorical

#######################################################
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
learning_rate = 2e-4
pretrained_weights = False
batch_size = 64   # default:64
epochs = 40     # default:40
# 'VIS-NIR': VIS-NIR patch dataset, 'OS': OS patch dataset, 'SEN1-2': SEN1-2 patch datset
choose_dataset = 'VIS-NIR'  # 'VIS-NIR', 'OS', 'SEN1-2'
# 'SCFDM':SCFDM, 'AFD':AFD-Net, 'MFD': MFD-Net, 'EFR':EFR-Net, 'FIL':FIL-Net, 'RRL':RRL-Net.
choose_model = 'RRL'   # choose one in ['SCFDM', 'AFD', 'MFD', 'EFR', 'FIL', 'RRL'].
input_model = access_model(choose_model)
out_dir_name = choose_model + '__' + choose_dataset
########################################################

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


root_path = os.path.abspath('.')
data_path = os.path.join(root_path, 'data')
time1 = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path, history_path, testout_path, fig_path, model_path = make_filedir(root_path, out_dir_name, time1)
best_model_path = os.path.join(model_path, 'match_model_best.hdf5')
best_model_test_path = os.path.join(model_path, 'match_model_best_test.hdf5')
new_name = "log" + time1 + ".txt"
sys.stdout = Logger(os.path.join(log_path, new_name))


if choose_model == 'SCFDM':
    matchnet = input_model()
    matchnet.compile(loss=['categorical_crossentropy', 'binary_crossentropy'],
                     optimizer=keras.optimizers.Nadam(lr=learning_rate),
                     metrics=['accuracy'])
elif choose_model == 'AFD':
    matchnet = input_model()
    matchnet.compile(loss='categorical_crossentropy',
                     optimizer=keras.optimizers.Nadam(lr=learning_rate),
                     metrics=['accuracy'])

elif choose_model == 'MFD' or choose_model == 'EFR':
    matchnet = input_model()
    matchnet.compile(loss='binary_crossentropy',
                     optimizer=keras.optimizers.Adam(lr=learning_rate),
                     metrics=['accuracy'])

elif choose_model == 'FIL':
    matchnet = input_model()
    matchnet.compile(loss='binary_crossentropy',
                     optimizer=keras.optimizers.Nadam(lr=learning_rate),
                     metrics=['accuracy'])

elif choose_model == 'RRL':
    matchnet, matchnet_test = input_model()
    matchnet.compile(loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy',
                           'binary_crossentropy', 'binary_crossentropy', p_mse_loss_01, p_mse_loss_01],
                     loss_weights=[1, 1, 1, 1, 1, 1, 1, 1],
                     optimizer=keras.optimizers.Nadam(lr=learning_rate),
                     metrics=['accuracy'])
    matchnet_test.compile(loss=['binary_crossentropy'],
                          loss_weights=[1],
                          optimizer=keras.optimizers.Nadam(lr=learning_rate),
                          metrics=['accuracy'])
else:
    print("Error!!!! Please input right model name!!!")
    sys.exit(0)


print("加载数据中........................................")
if choose_dataset == 'VIS-NIR':
    img = np.load(os.path.join(data_path, 'vis-nir',"country.npy"))
    label = np.load(os.path.join(data_path, 'vis-nir', "country_label.npy"))
    val_img = np.load(os.path.join(data_path, 'vis-nir', "val_data.npy"))
    val_label = np.load(os.path.join(data_path, 'vis-nir', "val_label.npy"))
elif choose_dataset == 'OS':
    img = np.load(os.path.join(data_path, 'os', "os_train_image.npy"))
    label = np.load(os.path.join(data_path, 'os', "os_train_label.npy"))
    val_img = np.load(os.path.join(data_path, 'os', "os_test_image.npy"))
    val_label = np.load(os.path.join(data_path, 'os', "os_test_label.npy"))
elif choose_dataset == 'SEN1-2':
    img = np.load(os.path.join(data_path, 'sen1-2', "sne_train_image.npy"))
    label = np.load(os.path.join(data_path, 'sen1-2', "sne_train_label.npy"))
    val_img = np.load(os.path.join(data_path, 'sen1-2', "sne_test_image.npy"))
    val_label = np.load(os.path.join(data_path, 'sen1-2', "sne_test_label.npy"))
else:
    print("Error!!!! Please input right dataset name!!!")
    sys.exit(0)


print("训练集图像集大小：", img.shape)
print("训练集标签集大小：", label.shape)
print("验证集图像集大小：", val_img.shape)
print("验证集标签集大小：", val_label.shape)

np.random.seed(100)
np.random.shuffle(img)
np.random.seed(100)
np.random.shuffle(label)

img0 = np.expand_dims(img[:, 0], axis=3)
img1 = np.expand_dims(img[:, 1], axis=3)

val_img0 = np.expand_dims(val_img[:, 0], axis=3)
val_img1 = np.expand_dims(val_img[:, 1], axis=3)

train_loss_mean_list = np.zeros(epochs)
train_acc_mean_list = np.zeros(epochs)
val_fpr95_list = np.zeros(epochs)
val_loss_list = np.zeros(epochs)
val_acc_list = np.zeros(epochs)

best_fpr95 = 100
best_i = 0

for i in range(0, epochs):
    epoch_model_name = "match_model" + str(i + 1) + ".hdf5"
    model_file_path = os.path.join(model_path, epoch_model_name)
    print("Epoch is {}/{}".format(i + 1, epochs))
    iters = int(len(label) / batch_size)
    train_loss_list = np.zeros(iters)
    train_acc_list = np.zeros(iters)
    for j in range(iters):
        rgb_aug, nir_aug, label_batch = gen4(img0, img1, label, batch_size)
        rgb_aug = rgb_aug / 255.0
        nir_aug = nir_aug / 255.0

        if choose_model == 'SCFDM':
            label_batch_multiclass = to_categorical(label_batch, num_classes=2)
            train_loss = matchnet.train_on_batch([rgb_aug, nir_aug],
                                                 [label_batch_multiclass, label_batch])
            train_loss_list[j] = train_loss[0]
            train_acc_list[j] = train_loss[3]
            print("   Epoch:{}/{},迭代次数：{}/{},训练loss:{:.4f},训练acc:{:.4f}".format(i + 1, epochs, j, iters, train_loss[0],
                                                                                train_loss[3]))

        elif choose_model == 'AFD':
            label_batch_multiclass = to_categorical(label_batch, num_classes=2)
            train_loss = matchnet.train_on_batch([rgb_aug, nir_aug],
                                                 [label_batch_multiclass, label_batch_multiclass])
            train_loss_list[j] = train_loss[0]  # 保存loss
            train_acc_list[j] = train_loss[3]  # 保存acc
            print("   Epoch:{}/{},迭代次数：{}/{},训练loss:{:.4f},训练acc:{:.4f}".format(i + 1, epochs, j, iters, train_loss[0],
                                                                                train_loss[3]))

        elif choose_model == 'MFD':
            train_loss = matchnet.train_on_batch(
                [rgb_aug, nir_aug, rgb_aug, nir_aug, rgb_aug, nir_aug, rgb_aug, nir_aug],
                [label_batch, label_batch, label_batch, label_batch, label_batch])
            train_loss_list[j] = train_loss[0]
            train_acc_list[j] = train_loss[6]
            print("   Epoch:{}/{},迭代次数：{}/{},训练loss:{:.4f},训练acc:{:.4f}".format(i + 1, epochs, j, iters, train_loss[0],
                                                                                train_loss[6]))

        elif choose_model == 'EFR':
            train_loss = matchnet.train_on_batch([rgb_aug, nir_aug, rgb_aug, nir_aug],
                                                 [label_batch, label_batch, label_batch])
            train_loss_list[j] = train_loss[0]
            train_acc_list[j] = train_loss[4]
            print("   Epoch:{}/{},迭代次数：{}/{},训练loss:{:.4f},训练acc:{:.4f}".format(i + 1, epochs, j, iters, train_loss[0],
                                                                                train_loss[4]))

        elif choose_model == 'FIL':
            train_loss = matchnet.train_on_batch([rgb_aug, nir_aug],
                                                 [label_batch, label_batch, label_batch, label_batch, label_batch,
                                                  label_batch])
            train_loss_list[j] = train_loss[0]
            train_acc_list[j] = train_loss[7]
            print("   Epoch:{}/{},迭代次数：{}/{},训练loss:{:.4f},训练acc:{:.4f}".format(i + 1, epochs, j, iters, train_loss[0],
                                                                                train_loss[7]))

        elif choose_model == 'RRL':
            train_loss = matchnet.train_on_batch([rgb_aug, nir_aug],
                                                 [label_batch, label_batch, label_batch, label_batch, label_batch,
                                                  label_batch, rgb_aug, nir_aug])
            train_loss_list[j] = train_loss[0]
            train_acc_list[j] = train_loss[9]
            print(
                "   Epoch:{}/{},迭代次数：{}/{},训练总loss:{:.4f},训练acc:{:.4f}" .format(i + 1, epochs, j, iters, train_loss[0], train_loss[9]))

        else:
            print("Error!!!! Please input right model name!!!")
            sys.exit(0)



    train_loss_mean_list[i] = np.mean(train_loss_list)
    train_acc_mean_list[i] = np.mean(train_acc_list)

    img_val_0 = val_img0 / 255.0
    img_val_1 = val_img1 / 255.0

    if choose_model == 'SCFDM':
        val_label_multiclass = to_categorical(val_label, num_classes=2)
        loss = matchnet.evaluate([img_val_0, img_val_1], [val_label_multiclass, val_label], verbose=1)
        val_loss_list[i] = loss[0]
        val_acc_list[i] = loss[3]

    elif choose_model == 'AFD':
        val_label_multiclass = to_categorical(val_label, num_classes=2)
        loss = matchnet.evaluate([img_val_0, img_val_1], [val_label_multiclass, val_label_multiclass], verbose=1)
        val_loss_list[i] = loss[0]
        val_acc_list[i] = loss[3]

    elif choose_model == 'MFD':
        loss = matchnet.evaluate(
            [img_val_0, img_val_1, img_val_0, img_val_1, img_val_0, img_val_1, img_val_0, img_val_1],
            [val_label, val_label, val_label, val_label, val_label], verbose=1)
        val_loss_list[i] = loss[0]
        val_acc_list[i] = loss[6]

    elif choose_model == 'EFR':
        loss = matchnet.evaluate(
            [img_val_0, img_val_1, img_val_0, img_val_1], [val_label, val_label, val_label], verbose=1)
        val_loss_list[i] = loss[0]
        val_acc_list[i] = loss[4]

    elif choose_model == 'FIL':
        loss = matchnet.evaluate([img_val_0, img_val_1],
                                 [val_label, val_label, val_label, val_label, val_label, val_label], verbose=1)
        val_loss_list[i] = loss[0]
        val_acc_list[i] = loss[7]

    elif choose_model == 'RRL':
        loss = matchnet.evaluate([img_val_0, img_val_1],
                                 [val_label, val_label, val_label, val_label, val_label, val_label, img_val_0, img_val_1],
                                 verbose=1)
        val_loss_list[i] = loss[0]
        val_acc_list[i] = loss[9]

    else:
        print("Error!!!! Please input right model name!!!")
        sys.exit(0)

    if choose_dataset == 'VIS-NIR':
        fpr_list = []
        for j in range(8):
            print("正在处理的类别为：", j)
            val0 = img_val_0[j * 10000:((j + 1) * 10000)]
            val1 = img_val_1[j * 10000:((j + 1) * 10000)]
            label_input = val_label[j * 10000:((j + 1) * 10000)]

            if choose_model == 'SCFDM' or choose_model == 'AFD':
                label_out_linshi = matchnet.predict([val0, val1], batch_size=64)
                label_out_linshi = label_out_linshi[0]
                label_out = label_out_linshi[:, 1]
            elif choose_model == 'MFD':
                label_out = matchnet.predict([val0, val1,val0, val1,val0, val1,val0, val1], batch_size=64)
                label_out = label_out[0]
            elif choose_model == 'EFR':
                label_out = matchnet.predict([val0, val1, val0, val1], batch_size=64)
                label_out = label_out[0]
            elif choose_model == 'FIL':
                label_out = matchnet.predict([val0, val1], batch_size=64)
                label_out = label_out[0]
            elif choose_model == 'RRL':
                label_out = matchnet_test.predict([val0, val1], batch_size=64)
                # label_out = label_out[0]'
            else:
                print("Error!!!! Please input right model name!!!")
                sys.exit(0)

            val_fpr, val_tpr, val_thresholds = metrics.roc_curve(label_input, label_out)
            val_fpr95 = float(interpolate.interp1d(val_tpr, val_fpr)(0.95))
            fpr_list.append(val_fpr95)
        val_fpr95_out = np.mean(fpr_list)
        val_fpr95_list[i] = val_fpr95_out
    else:
        if choose_model == 'SCFDM' or choose_model == 'AFD':
            label_out_linshi = matchnet.predict([img_val_0, img_val_1], batch_size=64)
            label_out_linshi = label_out_linshi[0]
            label_out = label_out_linshi[:, 1]
        elif choose_model == 'MFD':
            label_out = matchnet.predict([img_val_0, img_val_1, img_val_0, img_val_1, img_val_0, img_val_1, img_val_0, img_val_1], batch_size=64)
            label_out = label_out[0]
        elif choose_model == 'EFR':
            label_out = matchnet.predict([img_val_0, img_val_1, img_val_0, img_val_1], batch_size=64)
            label_out = label_out[0]
        elif choose_model == 'FIL':
            label_out = matchnet.predict([img_val_0, img_val_1], batch_size=64)
            label_out = label_out[0]
        elif choose_model == 'RRL':
            label_out = matchnet_test.predict([img_val_0, img_val_1], batch_size=64)
            # label_out = label_out[0]'
        else:
            print("Error!!!! Please input right model name!!!")
            sys.exit(0)
        val_fpr, val_tpr, val_thresholds = metrics.roc_curve(val_label, label_out)
        val_fpr95_out = float(interpolate.interp1d(val_tpr, val_fpr)(0.95))
        val_fpr95_list[i] = val_fpr95_out


    if val_fpr95_out <= best_fpr95:
        best_fpr95 = val_fpr95_out
        best_i = i + 1
        if choose_model == 'RRL':
            matchnet.save(best_model_path)
            matchnet_test.save(best_model_test_path)
        else:
            matchnet.save(best_model_path)

    print("Epoch:{},该epoch下训练的loss_mean:{:.4f},训练acc_mean:{:.4f},测试集fpr95:{:.4f},最好i:{},最好fpr95:{:.4f}".format(i + 1,
                                                                                                               np.mean(
                                                                                                                   train_loss_list),
                                                                                                               np.mean(
                                                                                                                   train_acc_list),
                                                                                                               val_fpr95_out,
                                                                                                               best_i,
                                                                                                               best_fpr95))
    print("\n")
    # matchnet.save(model_file_path)

print("各轮epoch下验证集的准确率：", val_fpr95_list)

with open(history_path, 'w') as f:
    f.write("各轮epoch下训练集的loss：" + str(train_loss_mean_list))
    f.write("\n")
    f.write("各轮epoch下训练集的acc：" + str(train_acc_mean_list))
    f.write("\n")
    f.write("各轮epoch下测试集的loss：" + str(val_loss_list))
    f.write("\n")
    f.write("各轮epoch下测试集的acc：" + str(val_acc_list))
    f.write("\n")
    f.write("各轮epoch下验证集的fpr95：" + str(val_fpr95_list))
    f.write("\n")
    f.close()