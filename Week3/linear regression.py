from sklearn.datasets import load_diabetes
import numpy as np
import matplotlib.pyplot as plt
diabetes=load_diabetes()  #糖尿病数据集，包含442行数据，10个属性值，包括年龄，性别
print(diabetes.keys())
data = diabetes.data      #442*10
target = diabetes.target    #Target是一年后患疾病的定量指标

X=data[:,:1]     #选取data第一列数据 use only one feature
y=target
X_train=X[:-20]
X_test=X[-20:]
y_train=y[:-20].reshape((-1,1))   #行数据变为列数据
y_test=y[-20:].reshape((-1,1))

#linear regression class
class linear(object):
    def __init__(self):
        self.w=None
        self.b=None

    def loss(self,X,y):
        num_feature=X.shape[1]
        num_train=X.shape[0]

        h=X.dot(self.w) + self.b
        loss=0.5*np.sum(np.square(h-y))/num_train  #cost function

        dw=X.T.dot((h-y))/num_train    #weight
        db=np.sum((h-y))/num_train     #bias

        return loss,dw,db
    def train(self,X,y,learn_rate=0.001,iters=10000):
        num_feature=X.shape[1]
        self.w=np.zeros((num_feature,1))
        self.b=0
        loss_list=[]

        for i in range(iters):
            loss,dw,db=self.loss(X,y)
            loss_list.append(loss)
            self.w+=-learn_rate*dw
            self.b+=-learn_rate*db

            if i%500==0:
                print('iters=%d,loss=%f'%(i,loss))
        return loss_list

    def predit(self,X_test):
        y_pred=X.dot(self.w)+self.b
        return y_pred
    pass

#####测试训练语句
classify=linear()
print('start')
loss_list=classify.train(X_train,y_train)
print('end')

#可视化训练结果
f=X_train.dot(classify.w)+classify.b
fig=plt.figure()

plt.subplot(211)
plt.scatter(X_train,y_train,color='black')
plt.scatter(X_test,y_test,color='blue')
plt.plot(X_train,f,color='red')
plt.xlabel('X')
plt.ylabel('y')

plt.subplot(212)
plt.plot(loss_list,color='blue')
plt.xlabel('epochs')
plt.ylabel('errors')
plt.show()






