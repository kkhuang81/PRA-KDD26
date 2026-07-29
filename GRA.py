import numpy as np 
import argparse

def AlphaCal(numAgentPerEthnicity, numEthnicity, alpha0, eps):
    alpha = np.array([alpha0] * numEthnicity)
    numAgents = int(numAgentPerEthnicity.sum())
    for k in range(1,numEthnicity):        
        ratio = numAgentPerEthnicity[k-1] * 1.0 / numAgentPerEthnicity[k]        
        r = (numAgentPerEthnicity[k-1] + numAgentPerEthnicity[k]) * 1.0 / numAgents
        tepk = alpha[k-1] * np.power(1.0 + 1.0 / ratio, 1.0 - alpha[k-1])        
        while alpha[k] >= 2*eps:                        
            if tepk * np.power(r, alpha[k-1] - alpha[k]) < alpha[k] * np.power(1.0 + ratio, 1.0 - alpha[k]):                
                alpha[k] -= eps
            else:
                break
    print(alpha)                
    return alpha


def ResourceAllocation(numAgentPerEthnicity, numAptPerBlock, alpha0, eps):
    numEthnicity = numAgentPerEthnicity.size
    numBlocks = numAptPerBlock.size
    alpha = AlphaCal(numAgentPerEthnicity, numEthnicity, alpha0, eps)    
    U = np.array([[0.0]*numBlocks]*numEthnicity)
    for l in range(numBlocks):        
        for _ in range(numAptPerBlock[l]):
            margain = [0.0]*numEthnicity
            for k in range(numEthnicity):
                margain[k] = np.power((U[k][l] + 1.0)/numAptPerBlock[l], alpha[k]) - np.power(U[k][l]/numAptPerBlock[l], alpha[k])                
            kstar = np.argmax(margain)             
            U[kstar][l] += 1.0
    return U

# Meaure the Inequality acorss blocks
def Inequality(numAgentPerEthnicity,U,type,beta):
    numEthnicity, numBlocks = U.shape
    inequal = np.array([0.0]*numBlocks)
    for b in range(numBlocks):
        dis = U[:,b] / numAgentPerEthnicity
        if type == 0:
            inequal[b] = Atkinson(dis, beta)
        elif type == 1:
            inequal[b] = Nash(dis, numAgentPerEthnicity)                                
        else:
            print('type error')
            exit(1)  
    print(inequal)                      
    return inequal.sum()

# Nash Welfare
def Nash(dis, numAgentPerEthnicity):    
    numAgents = int(numAgentPerEthnicity.sum())
    numEthnicity = dis.size
    sum = 1.0
    for eth in range(numEthnicity):        
        sum *= np.power(dis[eth], numAgentPerEthnicity[eth]*1.0 / numAgents)
    return sum        

# Atkinson Inequality
def Atkinson(dis, beta):    
    mu = np.average(dis)
    dis = np.power(dis, 1.0 - beta)
    mu1 = np.power(np.average(dis), 1.0 / (1.0 - beta))
    return 1.0 - mu1 / mu

###################### Input information ###############
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=int, default=0, help='the type of the inequality metrics')
parser.add_argument('--time', type=int, default=1, help='the number of tests')
parser.add_argument('--numAgents', type=int, default=1000, help='the number of Agents')
parser.add_argument('--alpha0', type=float, default=0.1, help='the first alpha value')
parser.add_argument('--eps', type=float, default=0.0001, help='the inteval in alpha calculation')
parser.add_argument('--beta', type=float, default=0.5, help='the parameter of Atinkson inequality')

args = parser.parse_args()
print("--------------------------")
print(args)

#Agent Info
numAgents = args.numAgents
ethic_prop = np.array([0.74, 0.135, 0.09, 0.034])
ethic_prop[::-1].sort()
numEthnicity = ethic_prop.size

# Block Info
numAptPerBlock = np.array([218,114,211,327,120])
numApts = int(numAptPerBlock.sum())
numBlocks = numAptPerBlock.size

# Preprocessing
numAgentPerEthnicity = ethic_prop * numAgents
numAgents = int(numAgentPerEthnicity.sum())

InequalScores = np.array([0.0]*args.time)
for t in range(args.time):
    # Resource Allocation
    U=ResourceAllocation(numAgentPerEthnicity,numAptPerBlock,args.alpha0,args.eps)    

    print(U)
    # Calculate Inequality
    InequalScores[t] = Inequality(numAgentPerEthnicity,U,args.type, args.beta)

print(InequalScores)