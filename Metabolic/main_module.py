import numpy as np
from scipy.integrate import solve_ivp
import gurobipy as gp
from gurobipy import GRB
import itertools
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_stoichiometry(mode = 'solved'):
    if mode == 'solved':
        return np.array([[-1,-1,-1,0,0,0],[0,0,1,0,-1,-1],[1,1,-1,-1,3,0]])
    else:
        return np.array([[-1,-1,-1,0,0,0],[0,0,1,0,-1,-1],[1,1,-1,-1,3,0],[-1,-1,1,1,-3,0]])

def get_param():

    v = [1, 10, 1, 1e-1, 1, 0.01]
    k = [10, 0, 0.1, 1e-5, 1, 0]
    lnk = [np.log(x) if x > 0 else -np.inf for x in k]
    return v, k, lnk

def get_propensity():

    # (X- kY)という書き方。Xが正、Yが負
    propensity = []
    
    propensity.append({'X':1,'ATP':-1,'ADP':1}) 
    propensity.append({'X':1,'ADP':1,'ATP':-1})
    propensity.append({'X':1,'ATP':1,'Y':-1,'ADP':-1})
    propensity.append({'ATP':1,'ADP':-1})
    propensity.append({'Y':1,'ADP':1,'ATP':-1})
    propensity.append({'Y':1})
        
    return propensity

def get_optimization_params():
    conc_lb = 1e-6
    FuncPieceError = 1e-5
    max_path_length = 4
    max_path_length = max(len([x for x in get_param()[2] if not np.isinf(x)]), max_path_length)
    return conc_lb, FuncPieceError, max_path_length

def flux(x,v=get_param()[0],k=get_param()[1],mode='solved'):
    
    J = np.zeros(len(v),)
    if mode == 'solved':
        X, Y, ATP, ADP = x[0], x[1], x[2], (1 - x[2])
    else:
        X, Y, ATP, ADP = x[0], x[1], x[2], x[3]
    
    J[0] = v[0]*(X*ADP - k[0]*ATP)
    J[1] = v[1]*(X*ADP - k[1]*ATP)
    J[2] = v[2]*(X*ATP - k[2]*Y*ADP)
    J[3] = v[3]*(ATP - k[3]*ADP)
    J[4] = v[4]*(Y*ADP - k[4]*ATP)
    J[5] = v[5]*(Y - k[5])

    return J


def model_equation(t,x,param):
    S = get_stoichiometry()

    if param['reverse']:
        J = -flux(x)
    else:
        if t > 1e1 and t < 1e3:
            v = param['v_next'][:]
            J = flux(x,v)
        else:
            J = flux(x)

    return S@J

# the main simulation module
def ComputeODE(param,init,tlist):
    sol = solve_ivp(model_equation, [0, np.max(tlist)], init, args=(param,), method='Radau',atol=1e-8, rtol=1e-6,t_eval=tlist)
    return sol

def different_bit(x,y):
    # return bit index where x and y are different
    return np.where(x != y)[0][0]

def compute_transitivity_single_path(source, target, flip_rxns, OutputFlag=0, ComputeIIS=False, conc_lb = 1e-10, FuncPieceError=1e-4):
    S = get_stoichiometry()
    lnk = get_param()[2]
    propensity = get_propensity()

    ChemName = ['X','Y','ATP','ADP']
    L = len(flip_rxns) + 1
    N, R = np.shape(S)

    # ====== sigmaは、軌道が辿る部分集合Wの符号。なのでL個 =====
    sigma = np.zeros((L,R),dtype=int)
    sigma[0,:] = np.sign(flux(source, np.ones(len(lnk)), [0 if np.isinf(_) else np.exp(_) for _ in lnk],mode='unsolved'))
    for i in range(1,L):
        sigma[i,:] = sigma[i-1,:].copy()
        sigma[i,flip_rxns[i-1]] *= -1
    
    # ======= reactions direction =======
    # reaction directionはWの内部（境界は含まない）に関する方向の情報。
    # x_{i-1}からx_iへの遷移Svの符号(積分の情報)を表すものをdir_v[i]とする。
    dir_v = np.copy(sigma)

    # ======= propensity direction ======
    # propensity direction制約は、sourceとtarget以外にかかり、null-reaction manifold上の点での反応の向き
    # x_iにかかる制約をdir_p[i]とする。
    dir_p = np.zeros((L,R))
    for i in range(L-1):
        dir_p[i,:] = np.copy(sigma[i,:])
        dir_p[i,flip_rxns[i]] = 0
    
    model = gp.Model()
    model.params.OutputFlag = OutputFlag 
    model.params.ScaleFlag = 0
    model.params.FuncNonlinear = 0
    model.params.FeasibilityTol = 1e-9
    model.params.OptimalityTol = 1e-9
    model.params.TimeLimit = 180
    #model.params.NumericFocus = 3
    
    x_ub = {'X':10.0, 'Y':300.0, 'ATP':1.0, 'ADP':1.0}
    lnx = {i:{m:model.addVar(vtype=GRB.CONTINUOUS, lb=np.log(conc_lb), ub=np.log(x_ub[m]), name=f'lnx{i}_{m}') for m in ChemName} for i in range(-1,L)}
    x = {i:{m:model.addVar(vtype=GRB.CONTINUOUS, lb=conc_lb, ub=x_ub[m], name=f'x{i}_{m}') for m in ChemName} for i in range(-1,L)}
    v = [[model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'v_step{i}_rxn{j}') for j in range(R)] for i in range(L)]
    xi =  {i:{m:model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f'slack_{m}_{i}_plus') for m in ChemName} for i in range(L)}
    eta =  {i:{m:model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f'slack_{m}_{i}_minus') for m in ChemName} for i in range(L)}
    inf_norm = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f'inf_norm')
    model.update()
    
    
    # objective
    for i in range(L):
        for m in ChemName:
            model.addLConstr(xi[i][m] <= inf_norm, name=f'inf_norm_{i}_{m}_plus')
            model.addLConstr(eta[i][m] <= inf_norm, name=f'inf_norm_{i}_{m}_minus')
    model.setObjective(inf_norm, GRB.MINIMIZE)

    # concentration constraint
    for i in x.keys():
        for m in ChemName:
            model.addGenConstrExp(lnx[i][m],x[i][m],name=f'exp{i}_{m}',options=f"FuncPieces=-2 FuncPieceError={FuncPieceError}") # x = exp(lnx)
    model.update()
    
    for i in x.keys():
        model.addLConstr(x[i]['ATP'] + x[i]['ADP'] == 1.0, name=f'conservation_{i}')
    
    # target and source
    for m in ChemName:
        model.addLConstr(x[-1][m] == source[ChemName.index(m)], name=f'source_{m}')
        model.addLConstr(x[L-1][m] == target[ChemName.index(m)], name=f'target_{m}')
        model.addLConstr(lnx[-1][m] == np.log(source[ChemName.index(m)]), name=f'source_{m}_log')
        model.addLConstr(lnx[L-1][m] == np.log(target[ChemName.index(m)]), name=f'target_{m}_log')
    

    # Connection Constraints
    for l in range(L):
        for n,m in enumerate(ChemName):
            if m == 'ADP':
                continue
            expr = gp.LinExpr()
            for r in range(R):
                expr += S[n,r]*v[l][r]
            model.addLConstr(expr + xi[l][m] - eta[l][m] == x[l][m] - x[l-1][m], name=f'connection_{l}_{m}')
    
    # flux direction constraint (v)
    for l in range(L):
        for r in range(R):
            if dir_v[l,r] > 0:
                model.addLConstr(v[l][r] >= 0, name=f'flux_{l}_{r}_plus')
            else:
                model.addLConstr(v[l][r] <= 0, name=f'flux_{l}_{r}_minus')
    
    # flux direction constraint (propensity) : ここの制約は、null-reaction manifold上の点での制約。上の制約はx_{i-1}からx_iへ移動するときのフラックスの積分への制約
    for l in range(L-1):
        for r in range(R):
            
            if np.isinf(lnk[r]):
                continue

            expr = gp.LinExpr()
            for m in propensity[r].keys():
                expr += lnx[l][m]*propensity[r][m] 
            expr -= lnk[r] 

            if dir_p[l,r] > 0:
                model.addLConstr(expr >= 0, name=f'propensity_{l}_{r}')
            elif dir_p[l,r] < 0:
                model.addLConstr(expr <= 0, name=f'propensity_{l}_{r}')
            elif dir_p[l,r] == 0:
                model.addLConstr(expr == 0, name=f'null_reaction_{l}_{r}')
            else:
                raise ValueError('invalid direction')
    
    model.update()
    model.optimize()
    
    if ComputeIIS:
        if model.status == 3:
            # compute IIS
            model.computeIIS()
            # show IIS
            for c in model.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)
    
    optimized_result = {}
    if model.status == 2:
        if OutputFlag:
            for l in x.keys():
                print([x[l][m].x for m in ChemName])

        optimized_result['x'] = [[x[l][m].x  for m in ChemName] for l in range(-1,L)]
        optimized_result['lnx'] = [[lnx[l][m].x  for m in ChemName] for l in range(-1,L)]
        optimized_result['v'] = [[v[l][r].x  for r in range(R)] for l in range(L)]
        objVal = model.objVal
    
    else:
        objVal = np.nan
    
    if model.status == 9:
        print('Time Limit exceeded')

    return model.status == GRB.OPTIMAL and model.objVal < 1e-3, model.status, objVal, optimized_result



def compute_transitivity(source, target, conc_lb, max_path_length,OutputFlag=0,FuncPieceError=1e-4,experr_tol=0.05):
    
    x, y, z = target[:3]
    if len(target) < 4:
        target = np.array([x,y,z,1-z])
    
    x, y, z = source[:3]
    if len(source) < 4:
        source = np.array([x,y,z,1-z])
    
    sigma_source = np.sign(flux(source,mode='unsolved'))
    sigma_target = np.sign(flux(target,mode='unsolved'))
    
    if np.sum(abs(sigma_source)) < 0.9*np.shape(sigma_source)[0]: #ちょうど0のときは除く
        return [[x,y,z],1e15]
    
    transitions = find_transitions(list(sigma_source), list(sigma_target), max_path_length)
    transitions = sorted(transitions, key=lambda _: len(_))
    min_length = hamming_distance(sigma_source, sigma_target)
    transitions = [_ for _ in transitions if len(_) >= min_length and (len(_) - min_length) % 2 == 0]
    
    min_violation = 1e10
    print(f'source = '+' '.join([str(_) for _ in source]))
    print(f'target = '+' '.join([str(_) for _ in target]))
    best_sol = {}
    timelimit_count = 0
    for flip_rxns in tqdm(transitions,total = len(transitions)):
        tmp = []
        val = compute_transitivity_single_path(source, target, flip_rxns, OutputFlag=OutputFlag, conc_lb = conc_lb, FuncPieceError=FuncPieceError)#, ComputeIIS=True)
        tmp.append([val[1],val[2]])
        if val[1] == 9:
            timelimit_count += 1 

        if min_violation > val[2]:
            min_violation = val[2]
            best_sol = val[3]

        if val[1] == 2:
            # if the experr is within the error tolerance and the violation is small enough, break the loop
            exp_error = [[abs(np.exp(val[3]['lnx'][l][m])-val[3]['x'][l][m])/np.average([np.exp(val[3]['lnx'][l][m]),val[3]['x'][l][m]]) for l in range(len(val[3]['x']))] for m in range(4)]
            exp_error = list(itertools.chain.from_iterable(exp_error))
            if val[2] < 1e-16 and np.max(exp_error) < experr_tol:
                break

    print(f'min_violation: {min_violation} at {x,y,z}')
    if best_sol == {}:
        print('exp error: avg -- max --')
        exp_error = [1]
    else:
        exp_error = [[abs(np.exp(best_sol['lnx'][l][m])-best_sol['x'][l][m])/np.average([np.exp(best_sol['lnx'][l][m]),best_sol['x'][l][m]]) for l in range(len(best_sol['x']))] for m in range(4)]
        exp_error = list(itertools.chain.from_iterable(exp_error))
        print(f'exp error: avg {np.average(exp_error)} max {np.max(exp_error)}')
    print()

    if len(flip_rxns) == 0:
        return [[x,y,z],max(0,min_violation),best_sol,-1,np.max(exp_error)]
    else:
        return [[x,y,z],max(0,min_violation),best_sol,timelimit_count/len(flip_rxns),np.max(exp_error)]
    

def find_different_bit_index(arr1, arr2):
    """ 2つの配列を比較して、異なるビットがあるインデックスを返す。
        2つ以上の異なるビットがある場合はエラーを返す。 """
    if len(arr1) != len(arr2):
        raise ValueError("配列の長さが異なります。")

    different_bit_indices = [i for i, (a, b) in enumerate(zip(arr1, arr2)) if a != b]

    if len(different_bit_indices) != 1:
        raise ValueError("2つ以上のビットが異なる or 同じ配列が渡されました。")

    return different_bit_indices[0] if different_bit_indices else None

def hamming_distance(x1, x2):
    """ ハミング距離を計算する関数 """
    return sum(int(a) != int(b) for a, b in zip(x1, x2))

def flip_bit(x, index):
    """ ベクトルの指定された位置のビットを反転する関数 """
    x[index] *= -1
    return x

def find_transitions(x1, x2, L):

    """ x1 から x2 への遷移列を深さ優先探索で探索する関数 """
    if hamming_distance(x1, x2) > L:
        return None

    def dfs(current, path, depth):
        lnk = get_param()[2]
        flippable_bit = [i for i in range(len(lnk)) if not np.isinf(lnk[i])]
        """ 深さ優先探索を行うヘルパー関数 """
        if depth > L:
            return
        if current == x2:
            result.append(path.copy())
        for i in flippable_bit:
            next_state = current.copy()
            flip_bit(next_state, i)
            dfs(next_state, path + [next_state], depth + 1)

    result = []
    dfs(x1, [x1], 0)
    
    transitions = []
    for r in result:
        transitions.append([find_different_bit_index(r[i], r[i+1]) for i in range(len(r)-1)])
    return transitions


def export_reuslt(result,max_path_length,conc_lb,experr_tol,prefix='WholeSpace'):
    os.makedirs('result', exist_ok=True)
    output = ''
    for r in result:
        best_sol = r[2]
        if best_sol == {}:
            max_err, avg_err = 0, 0
        else:
            exp_error = [[abs(np.exp(best_sol['lnx'][l][m])-best_sol['x'][l][m])/np.average([np.exp(best_sol['lnx'][l][m]),best_sol['x'][l][m]]) for l in range(len(best_sol['x']))] for m in range(4)]
            exp_error = list(itertools.chain.from_iterable(exp_error))
            max_err, avg_err = np.max(exp_error), np.average(exp_error)
        output += str(r[0][0]) + ' ' + str(r[0][1]) + ' ' + str(r[0][2]) + ' ' + str(r[1]) + f' {max_err} {avg_err} {r[3]}\n'
    with open(f'result/{prefix}_flip{max_path_length}_lb{int(np.log10(conc_lb))}_experrtol{experr_tol}.txt', 'w') as f:
        f.write(output)

def export_trajectory(result,suffix):
    output = ''
    for i,r in enumerate(result):
        if r[1] < 1e-9:
            for j in range(len(r[2]['x'])):
                output += ' '.join([str(r[2]['x'][j][_]) for _ in range(3)]) + f' {i}\n'
            output += '\n'

    with open(f'result/trajectory_{suffix}.txt','w') as fp:
        fp.write(output)

def plot_experr(result):
    # === Calculate the errors ===
    exp_error_list = {'max':[],'avg':[]}
    for r in result:
        if r[2] != {}:
            best_sol = r[2]
            exp_error = [[abs(np.exp(best_sol['lnx'][l][m])-best_sol['x'][l][m])/np.average([np.exp(best_sol['lnx'][l][m]),best_sol['x'][l][m]]) for l in range(len(best_sol['x']))] for m in range(4)]
            exp_error = list(itertools.chain.from_iterable(exp_error))
            exp_error_list['max'].append(np.max(exp_error))
            exp_error_list['avg'].append(np.average(exp_error))

    # === Plot the distribution separately ===
    colors = ['red','blue']
    # Plotting
    plt.figure(figsize=(10, 6))
    for i, key in enumerate(['max','avg']):
        plt.hist(exp_error_list[key], bins=50, color=colors[i], alpha=0.6)
    plt.xlabel('GenContrExp Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend(['max','avg'])
    plt.show()

def plot_maxviol(result):
    # === Calculate the violations with small value added ===
    violations = [np.log10(1e-12 + r[1]) for r in result]

    # === Plotting ===
    plt.figure(figsize=(10, 6))
    plt.hist(violations, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Log10(max. violation + 1e-12)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Violations')
    plt.grid(True)
    plt.show()
