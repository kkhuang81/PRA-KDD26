import numpy as np 
import argparse
import random

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


def ResourceAllocation(numAptPerBlock, alpha, pref, JobCap):
    print('numAptPerBlock: ', numAptPerBlock)
    numEthnicity = numAgentPerEthnicity.size
    numBlocks = numAptPerBlock.size
    numJob = JobCap.size
    U = np.array([[0.0]*numBlocks]*numEthnicity)
    for l in range(numBlocks):        
        for cl in range(numAptPerBlock[l]):
            margain = [0.0]*numEthnicity
            for k in range(numEthnicity):
                margain[k] = np.power((U[k][l] + 1.0)/numAptPerBlock[l], alpha[k]) - np.power(U[k][l]/numAptPerBlock[l], alpha[k])                
            while sum(margain) > 0:    ## once identified, it would break out this while loop  
                flag = False          
                kstar = np.argmax(margain) 
                margain[kstar] = 0.0
                if pref[kstar].sum() == 0: continue #break #continue
                for idx in range(pref[kstar].size):
                    if pref[kstar][idx] > 0 and JobCap[idx] > 0:
                        pref[kstar][idx] -= 1
                        JobCap[idx] -= 1
                        flag = True                            
                        break
                if flag:
                    #if U[kstar][l] > ??
                    U[kstar][l] += 1.0
                    break        
                #if sum(margain) == 0.0:
                    #print('Error', l, cl)
                    #exit(1)
    return U

# Meaure the Inequality acorss blocks
def Inequality(numAgentPerEthnicity,U,type,beta):
    numEthnicity, numBlocks = U.shape
    inequal = np.array([0.0]*numBlocks)
    for b in range(numBlocks):
        dis = U[:,b] / numAgentPerEthnicity
        print('dis: ',dis)
        if type == 0:
            inequal[b] = Atkinson(dis, beta)
        elif type == 1:
            inequal[b] = Nash(dis, numAgentPerEthnicity)                                
        else:
            print('type error')
            exit(1)  
    print('inequal: ',inequal)                      
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
parser.add_argument('--quota', type=float, default=0.5, help='the quota of Job assignment to each department')
parser.add_argument('--ratio', type=float, default=1.0, help='the ratio of the data')

args = parser.parse_args()
print("--------------------------")
print(args)

# the proportion of quota of each Job for each department
Job_QUOTA = args.quota # normally it should not be more than 30%, at most 0.5 in this application

Job_CAPA = 100  ## the number of applicants a Job accepts, i.e., how many employees A job can recommended to

## Max number of Jobs a applicant applies to
MAX_Job = 20  ##top-20 then


score_ = np.load('preference_scores.npy')
applicant_departments_ = np.load('protected_attributes.npy').astype(int)

applicantNum, JobNum = score_.shape

numapplicant, numJob = int(applicantNum*args.ratio), int(JobNum*args.ratio)
print('numapplicant: ', numapplicant, ' numJob: ', numJob)


InequalScores = np.array([0.0]*args.time)

for t in range(args.time):

    applicantIdx = random.sample(range(applicantNum), numapplicant)
    JobIdx = random.sample(range(JobNum), numJob)

    score = score_[np.ix_(applicantIdx,JobIdx)]
    applicant_departments = applicant_departments_[applicantIdx]


    #Agent Info
    #numAgentPerEthnicity = dept_cap 
    cnt = int(applicant_departments.sum())
    numAgentPerEthnicity = np.array([applicant_departments.shape[0]-cnt, cnt])
    numAgents = numAgentPerEthnicity.sum()
    numEthnicity = numAgentPerEthnicity.size


    ######### calculate alphas
    tep=[]
    for idx in range(numEthnicity):
        tep.append((numAgentPerEthnicity[idx],idx))
    tep1=sorted(tep,key=lambda x:x[0],reverse=True)   
    tep2 = np.array([a for (a,b) in tep1])
    alpha_tep = AlphaCal(tep2, numEthnicity, args.alpha0, args.eps) 
    alpha = np.array([0.0]*numEthnicity)
    for i in range(numEthnicity):
        idx = tep1[i][1]
        alpha[idx] = alpha_tep[i]
    del tep, tep1, tep2, alpha_tep

    # Block Info
    JobCap = np.array([Job_CAPA]*numJob)
    numAptPerBlock = np.array([JobCap.sum()])
    numApts = int(numAptPerBlock.sum())
    numBlocks = numAptPerBlock.size

    pref = np.array([[0]*numJob]*numEthnicity)


    cset = set()    # Set of Jobs

    applicant = -1
    for row in score:
        applicant += 1
        partition_idx = np.argpartition(row, -MAX_Job)[-MAX_Job:]
        Jobs = partition_idx[np.argsort(-row[partition_idx])]
        pref[applicant_departments[applicant]][Jobs] += 1

        for Job in Jobs:
            if Job == '':
                continue
            cset.add(Job)

    dept, crs = pref.shape
    for idxt in range(dept):
        for idxc in range(crs):
            pref[idxt][idxc] = min(int(Job_QUOTA*JobCap[idxc]), pref[idxt][idxc])


    U=ResourceAllocation(numAptPerBlock, alpha, pref, JobCap)    

    print('Assignment:\n', U)
    # Calculate Inequality
    InequalScores[t] = Inequality(numAgentPerEthnicity,U,args.type, args.beta)

print('numAgent: ', numAgentPerEthnicity)
print('final_score: ', np.mean(InequalScores))
