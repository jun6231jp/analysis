
# coding: utf-8

# In[2]:


from argparse import ArgumentParser
def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-c', '--cycle', type=int,default=10000,help='learning cycle(default 10000)')
    argparser.add_argument('-kn', '--knearest',action='store_true',help='K-nearest mode')
    argparser.add_argument('-rbf', '--rbfkernel',action='store_true',help='RBF mode')
    argparser.add_argument('-lsvm', '--linearsvm',action='store_true',help='linear SVM mode')
    argparser.add_argument('-ksvm', '--kernelsvm',action='store_true',help='kernel SVM mode')
    argparser.add_argument('-dl', '--deeplearning',action='store_true',help='deeplearning mode(default)')
    argparser.add_argument('-l', '--layer', type=int,default=5,help='layer num 2-9(default 5)')
    argparser.add_argument('-bs', '--bsratio', type=int,default=5,help='batch size = num of error kinds x bsratio(default 5)')
    argparser.add_argument('-wd', '--weightdecay', type=float,default=0.000001,help='weight decay coefficient(default 0.000001)')
    argparser.add_argument('-rd', '--ridge',action='store_true',help='enable ridge(default)')
    argparser.add_argument('-ls', '--lasso', action='store_true',help='enable lasso')
    argparser.add_argument('-s', '--sigmoid',action='store_true',help='default')
    argparser.add_argument('-r', '--relu', action='store_true',help='')
    argparser.add_argument('-adad', '--adadelta',action='store_true',help='default')
    argparser.add_argument('-adab', '--adabound',action='store_true',help='')
    argparser.add_argument('-adag', '--adagrad',action='store_true',help='')
    argparser.add_argument('-adaw', '--adamw',action='store_true',help='')
    argparser.add_argument('-adam', '--adam',action='store_true',help='')
    argparser.add_argument('-msgd', '--momentsgd',action='store_true', help='')
    argparser.add_argument('-rmsp', '--rmsprop',action='store_true',help='')
    argparser.add_argument('-amsg', '--amsgrad',action='store_true',help='')
    argparser.add_argument('-amsb', '--amsbound',action='store_true',help='')
    argparser.add_argument('-cmsgd', '--correctedmomentsgd',action='store_true',help='')
    argparser.add_argument('-nstr', '--nesterovag',action='store_true',help='')
    argparser.add_argument('-msv', '--msvag',action='store_true',help='')
    argparser.add_argument('-rmspg', '--rmspropgraves',action='store_true',help='')
    argparser.add_argument('-smo', '--smorms3',action='store_true',help='')
    argparser.add_argument('-sgd', '--sgd',action='store_true', help='')
    return argparser.parse_args()

args = get_option()
rep=args.cycle #学習繰り返し回数
bs_ratio=args.bsratio #バッチサイズ比率
decay=args.weightdecay #Ridge回帰の重み減衰係数
layer=args.layer #層数
if layer < 2:
    layer=2
elif layer > 9:
    layer=9
if args.lasso:
    regression="lasso"
else :
    regression="ridge"
if args.relu:
    func="relu"
else :
    func="sigmoid"
mode=0
if args.knearest:
    mode=1
elif args.rbfkernel:
    mode=2
elif args.linearsvm:
    mode=3
elif args.kernelsvm:
    mode=4
input_width=0;
DATA="/home/user/py/data/"
RES="/home/user/py/result/"
import os
os.makedirs(RES,exist_ok=True)
log = open(RES+"log.txt","w")

import chainer
import chainer.links as L
import chainer.functions as F
import chainer.computational_graph as c
import numpy as np
from numpy.random import randint
from tabulate import tabulate

#CSVを読み込んで教師データ作成
from chainer.datasets import TupleDataset
datas = []
labels = []
count=0
csv_file = open(DATA+"DATA.txt")
log.write("reading "+DATA+"DATA.txt"+"\n")
for line in csv_file:
    if input_width == 0:
        input_width = len(line.split(","))-1;
    if len(line.split(","))!=input_width+1:
        continue
    chklen=0
    for x in line.split(",")[0:input_width+1]:
        if len(x) == 0:
            chklen=1
    if chklen==0:
        data = np.array([np.float32(float(x)) for x in line.split(",")[0:input_width]])
        label = np.array(line.split(",")[input_width],np.int32)
        datas.append(data)
        labels.append(label)
csv_file.close()

#ラベルごとに集計
dataset=TupleDataset(datas,labels)
uniq_data=np.unique(labels , return_counts=True)
kinds=len(uniq_data[0])
labelids=uniq_data[0]
labeldist=uniq_data[1]
header=['Label_No','Count']
table=[labelids,labeldist]
result=tabulate(np.array(table).transpose(),header,tablefmt="grid")
log.write(result+"\n")
#重複するデータセットはラベルを小さい番号優先(labelでsortしてuniqueなdataの先頭行を取得)
index=np.argsort(labels,kind='quicksort',axis=0)
labelsort_datas=[]
labelsort_labels=[]
for i in range(index.size):
    labelsort_datas.append(datas[index[i]])
    labelsort_labels.append(labels[index[i]])
datasort_datas, index=np.unique(labelsort_datas, axis=0, return_index=True)
datasort_datas = []
datas=[]
labels=[]
original_labels=labelids
for i in range(index.size):
    datas.append(labelsort_datas[index[i]])
    labels.append(labelsort_labels[index[i]])
dataset=TupleDataset(datas,labels)
labelsort_datas=[]
labelsort_labels=[]
log.write("duplication removed\n")
#sortしてLabelをスタックしなおす
uniq_data=np.unique(labels , return_counts=True)
kinds=len(uniq_data[0])
labelids=uniq_data[0]
labeldist=uniq_data[1]
bs=kinds * bs_ratio
header=['Label_No','Unique Count']
table=[labelids,labeldist]
result=tabulate(np.array(table).transpose(),header,tablefmt="grid")
log.write(result+"\n")
log.write("batch size:"+str(bs)+"\n")
#ミニバッチサイズ+繰り返し数より元データが小さい場合はデータを繰り返して拡張する
if (bs+rep) > len(dataset):
    extend=int((bs+rep)/len(dataset))+1
else:
    extend=1
log.write("data extension:x"+str(extend)+"\n")
#Labelごとに行数が同じになるようにする
datas=[]
labels=[]
ext=[]
for i in range(kinds):
    ext.append(100/float(int(max(labeldist)/labeldist[i])))
    for j in range(len(dataset)):
        if dataset[j][1]==labelids[i] :
            for k in range(int(max(labeldist)/labeldist[i])):
                for l in range(extend):
                    datas.append(dataset[j][0])
                    labels.append(i)
conf=np.unique(labels , return_counts=True)
header=['Label_No','ExtendedCount','Probability']
table=[conf[0],conf[1],ext]
result=tabulate(np.array(table).transpose(),header,tablefmt="grid")
log.write(result+"\n")
dataset=TupleDataset(datas,labels)
log.write("data extended\n")
#データをシャッフルする
from chainer.datasets import split_dataset_random
dataset, null= split_dataset_random(dataset, len(dataset) , seed=0)
log.write("data randomized\n")


#データセット全体を 7 : 3 の比率でランダム分割し、学習用、検証用のデータセットとする
from chainer.datasets import split_dataset_random
n_train = int(len(dataset) * 0.7)
n_valid = int(len(dataset) * 0.3)
train, valid = split_dataset_random(dataset, n_train, seed=0)
train_len=len(train)
valid_len=len(valid)
log.write('Training dataset size:'+str(train_len)+"\n")
log.write('Validation dataset size:'+str(valid_len)+"\n")
train_datas=[]
train_labels=[]
valid_datas=[]
valid_labels=[]

for i in range(n_train):
    train_datas.append(train[i][0])
    train_labels.append(train[i][1])

for i in range(n_valid):
    valid_datas.append(valid[i][0])
    valid_labels.append(valid[i][1])

#メモリ開放
labelids=[]
labeldist=[]
train=[]
valid=[]
labelsort_datas=[]
labelsort_labels=[]
uniq_data=[]

#K-nearest mode
if mode == 1:
    from sklearn import datasets
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import accuracy_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import pandas as pd
    from pandas import plotting
    class NearestNeighbors(object):
        def __init__(self):
            self._train_data = None
            self._target_data = None
        def fit(self, train_data, target_data):
            self._train_data = train_data
            self._target_data = target_data
        def predict(self, x):
            distances = np.array([self._distance(p, x) for p in self._train_data])
            nearest_index = distances.argmin()
            return self._target_data[nearest_index]
        def _distance(self, p0, p1):
            return np.sum((p0 - p1) ** 2)
    predicted_labels = []
    loo = LeaveOneOut()
    log.write("learning and cross validation ..\n")
    for train,test in loo.split(datas):
        train_data = []
        target_data = []
        test_data = []
        for i in train:
            train_data.append(datas[i])
            target_data.append(labels[i])
        for j in test:
            test_data.append(datas[j])
        model = NearestNeighbors()
        model.fit(train_data, target_data)
        pred_label = model.predict(test_data[0])
        pred_labels.append(pred_label)
    accuracy = accuracy_score(labels, pred_labels)
    log.write("accuracy :"+str(accuracy)+"\n")

#RBF mode
elif mode == 2:
    from sklearn import linear_model, metrics
    from sklearn.kernel_approximation import RBFSampler
    rbf_feature = RBFSampler(gamma=1, n_components=input_width, random_state=1)
    clf_result=linear_model.SGDClassifier(loss="hinge")
    log.write("learning ...\n")
    clf_result.fit(train_datas, train_labels)
    log.write("validation ..\n")
    pred_labels=clf_result.predict(valid_datas)
    accuracy=metrics.accuracy_score(valid_labels,pred_labels)
    log.write("accuracy :"+str(accuracy)+"\n")

#Linear SVM mode
elif mode == 3:
    from sklearn.svm import SVC
    svc = SVC(kernel='linear', random_state=0)
    log.write("learning ...\n")
    svc.fit(train_datas, train_labels)
    from sklearn.metrics import accuracy_score
    log.write("validation ..\n")
    pred_labels = svc.predict(valid_datas)
    accuracy = accuracy_score(valid_labels, pred_labels)
    log.write("accuracy :"+str(accuracy)+"\n")

#Kernel SVM mode
elif mode == 4:
    from sklearn.svm import SVC
    svc = SVC(kernel='poly' , C=1.0,class_weight='balanced', random_state=0)
    log.write("learning ...\n")
    svc.fit(train_datas, train_labels)
    from sklearn.metrics import accuracy_score
    log.write("validation ..\n")
    pred_labels = svc.predict(valid_datas)
    accuracy = accuracy_score(valid_labels, pred_labels)
    log.write("accuracy :"+str(accuracy)+"\n")

#Deep Learning mode
else:
    #ニューラルネットワーク(NN)定義 (入力幅21,中間幅taper,出力幅エラー種類数)
    inputs=len(train_datas[0])
    unit_num=[0,0,0,0,0,0,0,0,0] #各層内のユニット数,最大9層
    for i in range(layer):
        unit_num[i]=int((inputs-kinds)*(layer-i)/layer+kinds)
    log.write("unit nums:"+str(unit_num)+"\n")
    from chainer import Chain
    class NN(Chain):
        def __init__(self,n_unit1=unit_num[0],n_unit2=unit_num[1],n_unit3=unit_num[2],
                     n_unit4=unit_num[3],n_unit5=unit_num[4],n_unit6=unit_num[5],
                     n_unit7=unit_num[6],n_unit8=unit_num[7],n_unit9=unit_num[8],
                     n_out=kinds):
            super(NN, self).__init__(
                fc1 = L.Linear(None, n_unit1),
                fc2 = L.Linear(None, n_unit2),
                fc3 = L.Linear(None, n_unit3),
                fc4 = L.Linear(None, n_unit4),
                fc5 = L.Linear(None, n_unit5),
                fc6 = L.Linear(None, n_unit6),
                fc7 = L.Linear(None, n_unit7),
                fc8 = L.Linear(None, n_unit8),
                fc9 = L.Linear(None, n_unit9),
                fout = L.Linear(None, n_out),
            )

        def __call__(self,x,y):
            return F.softmax_cross_entropy(self.fwd(x),y)

        def fwd(self,x):
            if args.relu: #活性化関数選択
                self.act=F.relu
            else :
                self.act=F.sigmoid
            h = self.act(self.fc1(x))
            h = self.act(self.fc2(h))
            if layer==2:
               return self.fout(h)
            h = self.act(self.fc3(h))
            if layer==3:
               return self.fout(h)
            h = self.act(self.fc4(h))
            if layer==4:
               return self.fout(h)
            h = self.act(self.fc5(h))
            if layer==5:
               return self.fout(h)
            h = self.act(self.fc6(h))
            if layer==6:
               return self.fout(h)
            h = self.act(self.fc7(h))
            if layer==7:
               return self.fout(h)
            h = self.act(self.fc8(h))
            if layer==8:
               return self.fout(h)
            h = self.act(self.fc9(h))
            if layer==9:
               return self.fout(h)

    # ネットワークを作成
    from chainer import optimizers, training
    net = NN()

    # 最適化手法選択
    if args.momentsgd:
        optimizer = optimizers.MomentumSGD()
    elif args.adamw:
        optimizer = optimizers.AdamW()
    elif args.adabound:
        optimizer = optimizers.AdaBound()
    elif args.rmsprop:
        optimizer = optimizers.RMSprop()
    elif args.adam:
        optimizer = optimizers.Adam()
    elif args.sgd:
        optimizer = optimizers.SGD()
    elif args.adagrad:
        optimizer = optimizers.AdaGrad()
    elif args.amsgrad:
        optimizer = optimizers.AMSGrad()
    elif args.amsbound:
        optimizer = optimizers.AMSBound()
    elif args.correctedmomentsgd:
        optimizer = optimizers.CorrectedMomentumSGD()
    elif args.nesterovag:
        optimizer = optimizers.NesterovAG()
    elif args.msvag:
        optimizer = optimizers.MSVAG()
    elif args.rmspropgraves:
        optimizer = optimizers.RMSpropGraves()
    elif args.smorms3:
        optimizer = optimizers.SMORMS3()
    else:
        optimizer = optimizers.AdaDelta() #0.81

    optimizer.setup(net)

    if args.lasso:
        #Lasso回帰でスパース化
        from chainer.optimizer_hooks import Lasso
        for param in net.params():
            if param.name != 'b':
                param.update_rule.add_hook(Lasso(decay))
    else :
        #Ridge回帰で過学習抑制
        from chainer.optimizer_hooks import WeightDecay
        for param in net.params():
            if param.name != 'b':
                param.update_rule.add_hook(WeightDecay(decay))

    #ミニバッチ学習
    from chainer import Variable
    gx=[]
    gy=[]
    for i in range(rep):
        sffindx = np.random.permutation(train_len)
        x=Variable(np.array(train_datas)[sffindx[i:(i+bs) if (i+bs) < train_len else train_len]])
        t=Variable(np.array(train_labels)[sffindx[i:(i+bs) if (i+bs) < train_len else train_len]])
        net.cleargrads()
        loss=net(x,t)
        log.write(str(i)+" "+str(loss.data)+"\n")
        gx.append(i)
        gy.append(loss.data)
        if loss.data!=0:
            loss.backward()
            optimizer.update()

    #学習過程をグラフ化
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    axes.plot(gx, gy)
    axes.set_title('Learn Progress')
    axes.set_xlabel('learn cycle')
    axes.set_ylabel('error')
    plt.tight_layout()
    fig.savefig(RES+"graph.png")

    #NN構造図作成
    import pydot
    import pathlib
    g = c.build_computational_graph(loss)
    name=RES+"NN.dot"
    with open(name,'w') as o:
        o.write(g.dump())
    (graph,) = pydot.graph_from_dot_file(name)
    graph.write_png(RES+"NN.png")

    #検査データでモデル精度検証
    xv = Variable(np.array(valid_datas))
    tv = net.fwd(xv)
    pred_labels = tv.data
    nrow, ncol = pred_labels.shape
    ok = 0
    for i in range(nrow):
        cls = np.argmax(pred_labels[i,:])
        if cls == valid_labels[i]:
            ok += 1
    accuracy=(ok * 1.0)/ nrow
    log.write("accuracy :" +str(ok)+"/"+str(nrow)+" = "+str(accuracy)+"\n")

    #モデル作成
    from chainer import serializers
    net.to_cpu()
    serializers.save_npz(RES+"net.npz", net)

log.write("end\n")
log.close()
