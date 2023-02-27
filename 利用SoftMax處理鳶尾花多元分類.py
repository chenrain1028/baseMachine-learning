#數據、矩陣處理套件numpy
import numpy as np
#繪圖處理套件matplotlib
import matplotlib.pyplot as plt
#繪圖處理套件顯示中文matplotlib.font_manager
import matplotlib.font_manager as plt_font
#設定中文字體物件和字型檔案路徑
twfont1=plt_font.FontProperties(fname="字型/kaiu.ttf")
from IPython import display
data=np.loadtxt("資料集/iris_data.csv",delimiter=",")
data_x=data[:,:4]
data_x=(data_x-data_x.mean(axis=0))/data_x.std(axis=0)
data_y=data[:,4]
#將分類標示改為one hot encoding
one_hot_y=np.zeros(( len(data_y) , 3 ) )
for i in range(len(data_y)):
  one_hot_y[i,int(data_y[i])]=1
#分割訓練資料集和測試資料集的特徵矩陣X、標籤矩陣Y
Train_X=data_x[:100]
Train_Y=one_hot_y[:100]     #這裡用ONE_hot的目的是為了計算損失函數
Test_X=data_x[100:]
Test_Y=data_y[100:]
#定義激活函數
def SoftMax(z):
  return np.exp(z)/np.sum(np.exp(z),axis=1,keepdims=True)       #為了保持維度不讓維度變回一維，後面要加keepdims
#定義模型運算函數
def F(X):
    z = np.dot(X, W) + B
    return SoftMax(z)
#定義損失函數
def Loss(Yh,Y):             #Yh預測值，Y實際值
  return -np.sum( Y*np.log(Yh) )
#設定訓練模型的參數
B=np.random.randn(3).reshape((-1,3))  #每個感知器都要有節點
W=np.random.randn(12).reshape((4,3) ) #特徵4輸出3故要12個權重
ETA=0.07
Step_W=[]
Step_B=[]
Step_L=[]
#用迴圈訓練模型
for epoch in range(2500):
  Step_B.append(B)
  Step_W.append(W)
  Yh=F(Train_X)
  W=W-ETA*np.dot(Train_X.T,Yh-Train_Y)
  B=B-ETA*np.sum(Yh-Train_Y)
  loss=Loss(Yh,Train_Y)
  Step_L.append(loss)
  if epoch%5==4:
    print("訓練次數",epoch+1,"損失值",loss)
    display.clear_output(wait=True)
#觀察訓練過程中的損失函數Loss變化
plt.figure(figsize=(9,6))
plt.title("Loss變化",fontproperties=twfont1,fontsize=20)
plt.xlabel("訓練次數",fontproperties=twfont1,fontsize=20)
plt.ylabel("損失函數",fontproperties=twfont1,fontsize=20)
plt.plot(Step_L,"b.",label="Loss")
plt.legend(prop=twfont1)
#測試資料集的準確度
TestYh=F(Test_X)
print("測試集的準確度",100*(np.argmax(TestYh,axis=1)==Test_Y).sum()/len(Test_Y),"%")