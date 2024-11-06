def systemSimulate(A,B,C,U,x0):
    import numpy as np
    simTime=U.shape[1]
    n=A.shape[0]
    r=C.shape[0]
    X=np.zeros(shape=(n,simTime+1))
    Y=np.zeros(shape=(r,simTime))
    for i in range(0,simTime):
        if i==0:
            X[:,[i]]=x0
            Y[:,[i]]=np.matmul(C,x0)
            X[:,[i+1]]=np.matmul(A,x0)+np.matmul(B,U[:,[i]])
        else:
            Y[:,[i]]=np.matmul(C,X[:,[i]])
            X[:,[i+1]]=np.matmul(A,X[:,[i]])+np.matmul(B,U[:,[i]])
    return Y,X

        

