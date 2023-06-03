from fastapi import FastAPI, File, Form, UploadFile
import numpy as np
import pandas as pd
import random
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
def process_excel(file:UploadFile = File(...), sij:float = Form(...)):
    data = pd.read_excel(file.file,header=None)  # Ganti nama sheet dengan yang sesuai
    n = len(data.columns)
    lj = data.iloc[n, :].values
    Cij = data.drop(n).values

    pSize = 5 * (n-1)
    sp = pSize // 2
    maxGen = 10 * n
    keep = pSize // 10
    runnumber = 10

    for rec in range(runnumber):
        p = np.ones((pSize, n))
        bestObj = np.ones(maxGen)
        for i in range(pSize):
            p[i,:] = np.random.permutation(n)+1  # random population of swarm
            
        cost = np.zeros(pSize)
        for s in range(pSize):
            cost[s] = calcost(p[s, :], Cij, lj, sij, n)
        
        ind = np.argmin(cost)
        gbest = cost[ind]
        bestSol = p[ind, :]
        for g in range(maxGen):
            newP = np.copy(p)
            sumCost = np.sum(cost)
            pf = cost / sumCost
            cumpf = np.cumsum(pf)
            
            for ip in range(0, pSize, 2):
                if random.random() <= 0.7:
                    child = np.zeros((2, n))
                    I = random.randint(0, n-3)
                    J = random.randint(0, n-2)
                    if I == J:
                        J = I + 1
                    
                    cps = min(I, J)
                    cpd = max(I, J)
                    
                    indf = np.searchsorted(cumpf - random.random(), 0)
                    indm = np.searchsorted(cumpf - random.random(), 0)
                    father = p[indf, :]
                    mother = p[indm, :]
                    child[0, cps:cpd] = mother[cps:cpd]
                    child[1, cps:cpd] = father[cps:cpd]
                    
                    restf = np.setdiff1d(father, child[0, :])
                    restm = np.setdiff1d(mother, child[1, :])

                    if len(restf) > 0:
                        if len(restf) == 1:
                            child[0, cpd] = restf[0]
                        else:
                            child[0, cpd:n] = restf[0:n-cpd]
                            child[0, 0:cps] = restf[n-cpd:]
                    
                    if len(restm) > 0:
                        if len(restm) == 1:
                            child[1, cpd] = restm[0]
                        else:
                            child[1, cpd:n] = restm[0:n-cpd]
                            child[1, 0:cps] = restm[n-cpd:]
                    
                    newP[ip, :] = child[0, :]
                    newP[ip+1, :] = child[1, :]
            
                if random.random() <= 0.3:
                    indp = np.argmax(cumpf > random.random())
                    par = p[indp].copy()
                    I = np.ceil(random.random() * (n - 2))
                    J = np.ceil(random.random() * (n - 1))
                    if I == J:
                        J = I + 1

                    k = np.ceil(random.random() * 3)
                    if k == 1:  # Flip
                        p[indp, int(I):int(J)] = np.flipud(par[int(I):int(J)])
                    elif k == 2:  # Swap
                        p[indp, [int(I),int(J)]] = par[[int(J),int(I)]]
                    elif k == 3:  # Slide
                        if I<J:
                            backup = p[indp, int(I)]
                            p[indp, int(I):int(J)] = par[int(I) + 1:int(J) + 1]
                            p[indp, int(J)] = backup
                        else :
                            backup = p[indp, int(J)]
                            p[indp, int(J):int(I)] = par[int(J) + 1:int(I) + 1]
                            p[indp, int(I)] = backup
            
            cost = np.zeros(pSize)
            for s in range(pSize):
                cost[s] = calcost(p[s], Cij, lj, sij, n)

            ind = np.argmin(cost)
            tmp = cost[ind]
            if tmp < gbest:
                gbest = tmp
                bestSol = p[ind].copy()
            bestObj[g] = gbest
            bestCurGen = bestSol.copy()

            for i in range(keep):
                if g == 0:
                    p[i] = bestSol.copy()
                    
                    cost[i] = gbest
                else:
                    ind2 = np.argmax(elitSol)
                    tmp2 = elitSol[ind2]

                    if tmp < tmp2:
                        elit[ind2] = bestCurGen.copy()
                        elitSol[ind2] = tmp

            if g == 0:
                elit = p[:keep].copy()
                elitSol = cost[:keep].copy()
            else:
                p[:keep] = elit.copy()


    res = []
    for i in bestSol.tolist():
        res.append(lj.tolist()[int(i-1)])
    return {'bestSol':res, 'origin': lj.tolist(), 'gap':sij, 'cost':gbest}

def calcost(F = None,Cij = None,lj = None,sij = None,n = None): 
    costs = 0
    for i in range(n-1):
        for j in range(i+1,n):
            if j == i + 1:
                dij = ((lj[int(F[i-1]-1)] + lj[int(F[j-1]-1)]) / 2) + sij
            else:
                tmp = 0
                for k in np.arange(i + 1,j - 1+1).reshape(-1):
                    tmp = tmp + lj[int(F[k-1]-1)] + sij
                
                dij = tmp + ((lj[int(F[i-1]-1)] + lj[int(F[j-1]-1)]) / 2) + sij
            costs = costs + Cij[int(F[i-1]-1),int(F[j-1]-1)] * dij
    
    return costs