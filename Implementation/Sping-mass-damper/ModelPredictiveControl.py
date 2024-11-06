import numpy as np

class ModelPredictiveControl(object):
    def __init__(self,A,B,C,f,v,W3,W4,x0,desiredControlTrajectoryTotal):
        self.A=A #system matrices
        self.B=B
        self.C=C
        self.f=f #prediction horizon 
        self.v=v #control horizon
        self.W3=W3 #input weight matrix
        self.W4=W4 #prediction weight matix
        self.desiredControlTrajectoryTotal=desiredControlTrajectoryTotal

        #dimensions of matrices
        self.n=A.shape[0]
        self.r=C.shape[0]
        self.m=B.shape[1]

        self.currentTimeStep=0

        self.states=[]
        self.states.append(x0)

        self.inputs=[]

        self.outputs=[]
        self.outputs.append(np.matmul(C,x0))

        self.O, self.M, self.gainMatrix = self.formLiftedMatrices()

    def formLiftedMatrices(self):
        f=self.f
        v=self.v
        r=self.r
        n=self.n
        m=self.m
        A=self.A
        B=self.B
        C=self.C

        O=np.zeros(shape=(f*r,n))

        for i in range(f):
            if (i == 0):
                powA=A
            else:
                powA=np.matmul(powA,A)
            O[i*r:(i+1)*r,:]=np.matmul(C,powA)

        M=np.zeros(shape=(f*r,v*m))

        for i in range(f):
            if(i<v):
                for j in range(i+1):
                    if(j==0):
                        powA=np.eye(n,n)
                    else:
                        powA=np.matmul(powA,A)
                    M[i*r:(i+1)*r,(i-j)*m:(i-j+1)*m]=np.matmul(C,np.matmul(powA,B))
            else:
                for j in range(v):
                    if j==0:
                        sumLast=np.zeros(shape=(n,n))
                        for s in range(i-v+2):
                            if(s == 0):
                                powA=np.eye(n,n)
                            else:
                                powA=np.matmul(powA,A)
                            sumLast=sumLast+powA
                        M[i*r:(i+1)*r,(v-1)*m:(v)*m]=np.matmul(C,np.matmul(sumLast,B))
                    else:
                        powA=np.matmul(powA,A)
                        M[i*r:(i+1)*r,(v-1-j)*m:(v-j)*m]=np.matmul(C,np.matmul(powA,B))

        tmp1=np.matmul(M.T,np.matmul(self.W4,M))
        tmp2=np.linalg.inv(tmp1+self.W3)
        gainMatrix=np.matmul(tmp2,np.matmul(M.T,self.W4))

        return O,M,gainMatrix

    def propogateDynamics(self, controlInput, state):
        xkp1=np.zeros(shape=(self.n,1))
        yk=np.zeros(shape=(self.r,1))
        xkp1=np.matmul(self.A,state)+np.matmul(self.B,controlInput)
        yk=np.matmul(self.C,state)

        return xkp1,yk
    
    def computeControlInputs(self):
        desiredControlTrajectory=self.desiredControlTrajectoryTotal[self.currentTimeStep:self.currentTimeStep+self.f,:]

        vectorS=desiredControlTrajectory-np.matmul(self.O,self.states[self.currentTimeStep])

        inputSequenceComputed=np.matmul(self.gainMatrix,vectorS)
        inputApplied=np.zeros(shape=(1,1))
        inputApplied[0,0]=inputSequenceComputed[0,0]

        state_kp1,output_k=self.propogateDynamics(inputApplied,self.states[self.currentTimeStep])

        self.states.append(state_kp1)
        self.outputs.append(output_k)
        self.inputs.append(inputApplied)

        self.currentTimeStep=self.currentTimeStep+1



            
