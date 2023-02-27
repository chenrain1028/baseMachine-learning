#數據、矩陣處理套件numpy
import numpy as np
#繪圖處理套件matplotlib
import matplotlib.pyplot as plt
#繪圖處理套件顯示中文matplotlib.font_manager
import matplotlib.font_manager as plt_font
#設定中文字體物件和字型檔案路徑
twfont1=plt_font.FontProperties(fname="字型/kaiu.ttf")

data=np.loadtxt("資料集/iris_data.csv",delimiter=",")  #delimiter指定分隔符，默认为任何空格字符。

print(data.shape)   #用shape看有無全部抓入


plt.figure(figsize=(8,5)) #繪圖尺寸
plt.title("鳶尾花品種分佈圖",fontproperties=twfont1,fontsize=20)
plt.xlabel("花辦長度",fontproperties=twfont1,fontsize=20)
plt.ylabel("花辦寬度",fontproperties=twfont1,fontsize=20)
plt.scatter(data[ data[:,4]==0 ][:,2],data[data[:,4]==0][:,3],c="b",label="山鳶尾")#取出所有山鳶尾的"花辦長度"[所有列,所有欄位]
plt.scatter(data[ data[:,4]==1 ][:,2],data[data[:,4]==1][:,3],c="r",label="雜色鳶尾")#第一個為列，第二個為行索引
plt.scatter(data[ data[:,4]==2 ][:,2],data[data[:,4]==2][:,3],c="g",label="維吉尼亞鳶尾")
plt.legend(prop=twfont1)
plt.show()

plt.figure(figsize=(8,5)) #繪圖尺寸
plt.title("鳶尾花品種分佈圖",fontproperties=twfont1,fontsize=20)
plt.xlabel("花萼長度",fontproperties=twfont1,fontsize=20)
plt.ylabel("花萼寬度",fontproperties=twfont1,fontsize=20)
plt.scatter(data[ data[:,4]==0 ][:,0],data[data[:,4]==0][:,1],c="b",label="山鳶尾")#取出所有山鳶尾的"花萼長度"[所有列,所有欄位]
plt.scatter(data[ data[:,4]==1 ][:,0],data[data[:,4]==1][:,1],c="r",label="雜色鳶尾") #第一個條件篩一次，第二個條件在篩一次
plt.scatter(data[ data[:,4]==2 ][:,0],data[data[:,4]==2][:,1],c="g",label="維吉尼亞鳶尾")
plt.legend(prop=twfont1)
plt.show()