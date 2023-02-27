import torch

#產生特徵矩陣[0.0 , 1.0 , 2.0]，設定計算圖起點
X=torch.tensor([0.0,1.0,2.0],requires_grad=True)    #有設定requires_grad的話會自動追蹤每一個計算過程
print(X)

#計算預估值Yh=3X^2-3
Yh=3*X**2-3
Yh.retain_grad()    #前面有追蹤梯度所以要retain_grad
print(Yh)

#設定實際值矩陣Y=[2.0 , 4.0 , 6.0]，並計算誤差值E=Yh-Y
Y=torch.tensor([2.0 , 4.0 , 6.0])       #這部分沒有要追蹤他的梯度所以就不用requires_grad
E=Yh-Y
E.retain_grad()
print(E)

#計算誤差平方Z=E2
Z=E**2
Z.retain_grad()
print(Z)

#計算損失函數L=0.5ΣZ，為計算圖終點
L=0.5*Z.sum()
L.retain_grad()
print(L)

#反向求梯度
L.backward()
print("L對X的梯度",X.grad )
print("L對Z的梯度",Z.grad )
print("L對E的梯度",E.grad )
print("L對Yh的梯度",Yh.grad )
