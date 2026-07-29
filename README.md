## Parameterized Fair Resource Allocation under Diversity Constraints

This repository contains the official implementation of our KDD 2026 paper **Parameterized Fair Resource Allocation under Diversity Constraints**. 

We propose **PRA**, a parameterized framework for fair resource allocation under diversity constraints. Unlike existing methods that rely on rigid hard constraints, PRA introduces inequality-aversion parameters to flexibly control group fairness while maximizing social welfare. We further develop **APRA** to handle additional application-specific constraints. For further details, please refer to our paper in **KDD 2026** (https://arxiv.org/abs/xxxx.xxxx). Should you encounter any issues, please reach out to Keke Huang. Thanks!


## Code & Data
The code folder contains the source code of PRA/APRA for the three applications. Some data are already provided in the code. The data folder contains the raw data of the course assignment application.


## Execution Command

python GRA.py --type  --time  --numAgents --alpha0  --eps 

--type: 0 indicates Atkinson inequality; 1 indicates Nash welfare

--time: the repeated number

--alpha0: the initial value of $\alpha_1 \in (0,1)$

--epsilon: the stride parameter $\epsilon$

