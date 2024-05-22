
import numpy as np
import plotly.graph_objects as go
import model_module_glyc as mm
from scipy.spatial import ConvexHull
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
from scipy.optimize import fsolve
import networkx as nx
import matplotlib.pyplot as plt

def initialize_2d(plot_range):
    plt.figure(figsize=(6,6))
    plt.xlim(np.log10(plot_range['xmin']), np.log10(plot_range['xmax']))
    plt.ylim(np.log10(plot_range['ymin']), np.log10(plot_range['ymax']))
    plt.xlabel('x')
    plt.ylabel('y')
    # 凡例の表示
    plt.legend()
    # グリッドの表示
    plt.grid(True)
    # 描画範囲の設定
    x_values = np.linspace(np.log10(plot_range['xmin']), np.log10(plot_range['xmax']), 100)
    return x_values

def add_convex_hull_2d(filename):
    points_x, points_y, points_z = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            x, y, z, violation, max_err, avg_err, _ = map(float, line.split())
            if violation > 1e-9 and z > 10**-1.72:
                points_x.append(np.log10(x) + 1e-6*abs(np.log10(z)))
                points_y.append(np.log10(y) + 1e-6*abs(np.log10(z)))
                points_z.append(np.log10(z))
    points = np.array([points_x, points_y, points_z]).T
    points = points[:,:2]

    # 凸包を計算
    hull = ConvexHull(points)

    # 点をプロット
    plt.plot(points[:, 0], points[:, 1], 'o')

    # 凸包の境界を描画
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')


def add_null_manifold_2d(x_values, logz):

    k = mm.get_param()[1]
    # x = const (垂直線)
    x_const = np.log10(k[0]*10**logz/(1-10**logz))
    plt.axvline(x=x_const, color='r', linestyle='--', label='M0')

    # g(x) = x (対角線)
    plt.plot(x_values, np.log10(10**x_values*10**logz/(k[2]*(1-10**logz))), linestyle='--',label='M2')

    # y = const (水平線)
    y_const = np.log10(k[4]*10**logz/(1-10**logz))
    plt.axhline(y=y_const, color='g', linestyle='--', label='M4')


def add_boundary_2d(plot_range, goal, logz):

    # ===== boundary of bottom region =====
    logy = np.linspace(np.log10(plot_range['ymin']), np.log10(plot_range['ymax']), 64)
    x_bnd0 = np.log10(goal[0] + goal[2] - 10**logz + 2*(goal[1] - 10**logy))  
    x_bnd1 = np.log10(goal[0] + goal[2] - 10**logz + 3*(goal[1] - 10**logy))
    # ===== plot =====
    plt.plot(x_bnd0, logy, linestyle='--', color='cyan')
    plt.plot(x_bnd1, logy, linestyle='--', color='orange')

    # ===== boundary of top region =====
    logx = np.linspace(np.log10(plot_range['xmin']), np.log10(plot_range['xmax']), 64)
    y_bnd0 = np.log10(goal[1] + (goal[2] - 10**logz + goal[0] - 10**logx)/2)  
    y_bnd1 = np.log10(goal[1] + (goal[2] - 10**logz + goal[0] - 10**logx)/3)
    # ===== plot =====
    plt.plot(logx, y_bnd0, linestyle='--', color='cyan')
    plt.plot(logx, y_bnd1, linestyle='--', color='orange')


def add_trajectories(fig, filename):

    colors = [
        '#1f77b4',  # 明るい青
        '#ff7f0e',  # 明るいオレンジ
        '#2ca02c',  # 明るい緑
        '#d62728',  # 明るい赤
        '#9467bd',  # 明るい紫
        '#8c564b',  # 明るい茶色
        '#e377c2',  # 明るいピンク
        '#7f7f7f',  # 明るいグレー
        '#bcbd22',  # 明るい黄緑
        '#17becf'   # 明るいシアン
    ]


    points_x, points_y, points_z = defaultdict(list), defaultdict(list), defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            if line == '\n':
                continue
            x, y, z, idx = map(float, line.split())
            
            points_x[idx].append(np.log10(x))
            points_y[idx].append(np.log10(y))
            points_z[idx].append(np.log10(z))

    for idx in [int(_) for _ in points_x.keys()]:
        fig.add_trace(go.Scatter3d(x=points_x[idx], y=points_y[idx], z=points_z[idx],
                                mode='lines+markers', 
                                line=dict(color=colors[idx%len(colors)], width=2),
                                marker=dict(size=2, color=colors[idx%len(colors)]),  # 点の設定
                                name=f'Trajectory {idx}'))  # 凡例の名前
        
    return fig



def initialize(plot_range):
    fig = go.Figure()
    fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        # Adjust aspect ratio here:
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1)  # Example: Increase y-axis depth appearance
    ),
    width=1200,
    height=800
    )
    
    x = np.linspace(np.log10(plot_range['xmin']), np.log10(plot_range['xmax']), 64)
    y = np.linspace(np.log10(plot_range['ymin']), np.log10(plot_range['ymax']), 64)
    x, y = np.meshgrid(x, y)

    # ===== updating layout =====
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[np.log10(plot_range['xmin']), np.log10(plot_range['xmax'])],showgrid=True, showbackground=True, zeroline=True, showticklabels=True), 
            yaxis=dict(range=[np.log10(plot_range['ymin']), np.log10(plot_range['ymax'])],showgrid=True, showbackground=True, zeroline=True, showticklabels=True),  
            zaxis=dict(range=[np.log10(plot_range['zmin']), np.log10(plot_range['zmax'])],showgrid=True, showbackground=True, zeroline=True, showticklabels=True,autorange=False)   
        ),
        font=dict(
        family="Times New Roman",
        size=18,
        color="black"
        ),
        
    )

    return fig, x, y

def add_intersection(fig,plot_range):

    k = mm.get_param()[1]

    # intersection
    x = np.linspace(-6, 3, 256)
    y = np.log10((10**x)**2/k[0]/k[2])
    z = np.log10((10**x)/(k[0]+10**x))
    mask = (x < np.log10(plot_range['xmax'])) & (x > np.log10(plot_range['xmin'])) & (y < np.log10(plot_range['ymax'])) & (y > np.log10(plot_range['ymin'])) & (z < np.log10(plot_range['zmax'])) & (z > np.log10(plot_range['zmin']))
    x, y, z = x[mask], y[mask], z[mask]
    # plot intersection as line
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                            mode='lines', 
                            line=dict(color='black', width=5),  # 線の設定
                            name='P0 and P2'))  # 凡例の名前

    # intersection
    y = np.linspace(-6, 3, 256)
    x = np.array([np.log10(k[2]*k[4]) for _ in y])
    z = np.array([np.log10((k[2]*10**ye)/(10**xe+k[2]*10**ye)) for xe, ye in zip(x, y)])
    mask = (x < np.log10(plot_range['xmax'])) & (x > np.log10(plot_range['xmin'])) & (y < np.log10(plot_range['ymax'])) & (y > np.log10(plot_range['ymin'])) & (z < np.log10(plot_range['zmax'])) & (z > np.log10(plot_range['zmin']))
    x, y, z = x[mask], y[mask], z[mask]
    # plot intersection as line
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                            mode='lines', 
                            line=dict(color='black', width=5, dash='dashdot'),  # 線の設定
                            name = 'P2 and P4'))  # 凡例の名前


    # intersection
    x = np.linspace(-6, 3, 256)
    y = np.log10(k[4]/k[0]*10**x)
    z = np.log10(10**y/(10**y+k[4]))
    # plot intersection as line
    mask = (x < np.log10(plot_range['xmax'])) & (x > np.log10(plot_range['xmin'])) & (y < np.log10(plot_range['ymax'])) & (y > np.log10(plot_range['ymin'])) & (z < np.log10(plot_range['zmax'])) & (z > np.log10(plot_range['zmin']))
    x, y, z = x[mask], y[mask], z[mask]
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                            mode='lines',  # 点と線の両方を表示
                            line=dict(color='black', width=5, dash='dash'),  # 線の設定
                            name='P4 and P0'))  # 凡例の名前

    return fig


def add_convex_hull(fig,points):

    points = np.array(points).T
    # add small noise one-by-one depending on z value
    for i in range(np.shape(points)[0]):
        points[i,0] += 1e-5*abs(points[i,2])
        points[i,1] += 1e-6*abs(points[i,2])
    
    hull = ConvexHull(points)

    # ConvexHullの各面をプロット
    for simplex in hull.simplices:
        x_hull = [points[simplex[i], 0] for i in range(3)] + [points[simplex[0], 0]]
        y_hull = [points[simplex[i], 1] for i in range(3)] + [points[simplex[0], 1]]
        z_hull = [points[simplex[i], 2] for i in range(3)] + [points[simplex[0], 2]]
        fig.add_trace(go.Mesh3d(x=x_hull, y=y_hull, z=z_hull, color='#C8C8C8', opacity=0.5))

    return fig

def add_attractors(fig,filename):
    X = np.loadtxt(filename,delimiter=',')
        
    death_attractor, live_attractor = X[1], X[0]

    live_attractor = [np.log10(_) for _ in live_attractor[:3]]
    death_attractor = [np.log10(_) for _ in death_attractor[:3]]

    fig.add_trace(go.Scatter3d(x=[live_attractor[0]], y=[live_attractor[1]], z=[live_attractor[2]],
                            mode='markers',
                            marker=dict(
                                size=4,               # 追加する点のサイズ
                                color='red',        # 追加する点の色
                                symbol='circle',      # 点の形状
                                opacity=1.0           # 透明度
                            ),
                            name='Live'  # 追加する点の名前（凡例に表示される）
    ))


    fig.add_trace(go.Scatter3d(x=[death_attractor[0]], y=[death_attractor[1]], z=[death_attractor[2]],
                            mode='markers',
                            marker=dict(
                                size=4,               # 追加する点のサイズ
                                color='blue',        # 追加する点の色
                                symbol='square',      # 点の形状
                                opacity=1.0          # 透明度
                            ),
                            name='Death'  # 追加する点の名前（凡例に表示される）
    ))

    return fig

def add_non_returnable_points(fig,plot_range,filename,pointsize=1):

    points_x, points_y, points_z = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            x, y, z, violation = map(float, line.split()[:4])
            if violation > 1e-9 and x < plot_range['xmax'] and x > plot_range['xmin'] and y < plot_range['ymax'] and y > plot_range['ymin'] and z < plot_range['zmax'] and z > plot_range['zmin']:
                points_x.append(np.log10(x))
                points_y.append(np.log10(y))
                points_z.append(np.log10(z))
    
    fig.add_trace(go.Scatter3d(x=points_x, y=points_y, z=points_z,
                            mode='markers',
                            marker=dict(
                                size=pointsize,               # 追加する点のサイズ
                                color='black',        # 追加する点の色
                                symbol='circle',      # 点の形状
                                opacity=1.0           # 透明度
                            ),
                            name='Non-returnable points'  # 追加する点の名前（凡例に表示される）
    ))
    return fig, [points_x,points_y,points_z]


def add_nulll_manifolds(fig,x0,y0,plot_range):

    k = mm.get_param()[1]
    
    x = x0.copy()
    y = y0.copy()
    z = np.log10(10**x/(k[0]+10**x))
    # f(x, y)のサーフェスプロット（赤色）
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.3, name='f(x,y)',
                            colorscale=[(0, '#FA8072'), (1, '#FA8072')], showscale=False))

    x = x0.copy()
    y = y0.copy()
    z = np.log10(k[2]*10**y/(10**x+k[2]*10**y))
    # g(x, y)のサーフェスプロット（青色）
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.3, name='g(x,y)',
                            colorscale=[(0, '#008080'), (1, '#008080')], showscale=False))

    x = x0.copy()
    y = y0.copy()
    z = np.log10(10**y/(10**y+k[4]))
    # h(x, y)のサーフェスプロット（緑色）
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.3, name='h(x,y)',
                            colorscale=[(0, '#6B8E23'), (1, '#6B8E23')], showscale=False))

    return fig



def get_arrow_list(starts, colors, model='linear'):
    v, k = mm.get_param()
    S = mm.get_stoichiometry()

    starts = [np.array(x) for x in starts]

    arrows = []
    for p in starts:
        if model == 'linear':
            sigma = np.sign(mm.flux(10.0**p, v, k))
        elif model == 'cubic':
            sigma = np.sign(mm.flux_cubic(10.0**p, v, k))
        for r in range(np.shape(S)[1]):
            norm = np.linalg.norm(S[:,r])
            end_point = sigma[r]*S[:,r]/norm*0.5 + p
            arrows.append([p, end_point, colors[r]])
    return arrows

def add_arrow(fig, arrow):
    arrow_start, arrow_end, color = arrow[0], arrow[1], arrow[2]
    
    # 矢印本体の描画
    fig.add_trace(go.Scatter3d(x=[arrow_start[0], arrow_end[0]], 
                               y=[arrow_start[1], arrow_end[1]], 
                               z=[arrow_start[2], arrow_end[2]], 
                               mode='lines', 
                               line=dict(color=color, width=5), showlegend=False))

    arrow_direction = arrow_end - arrow_start
    arrow_length = np.linalg.norm(arrow_direction)
    
    # 0除算を避けるためのチェック
    if arrow_length == 0:
        return fig
    
    arrow_direction /= arrow_length

    # 矢印の先端のサイズと角度
    arrowhead_length = 0.2
    arrowhead_angle = np.pi / 6

    # z軸に平行な場合の処理
    if abs(arrow_direction[0]) < 1e-20 and abs(arrow_direction[1]) < 1e-20:
        # x軸またはy軸に平行なベクトルを使う
        perpendicular_direction = np.array([1, 0, 0]) if arrow_direction[2] != 0 else np.array([0, 1, 0])
    else:
        # z軸に平行でない場合の通常の処理
        perpendicular_direction = np.array([-arrow_direction[1], arrow_direction[0], 0])

    left = arrow_end - arrowhead_length * (np.cos(arrowhead_angle) * arrow_direction + np.sin(arrowhead_angle) * perpendicular_direction)
    right = arrow_end - arrowhead_length * (np.cos(arrowhead_angle) * arrow_direction - np.sin(arrowhead_angle) * perpendicular_direction)

    # 矢印の先端（"V"字型）の描画
    fig.add_trace(go.Scatter3d(x=[left[0], arrow_end[0], right[0]], 
                               y=[left[1], arrow_end[1], right[1]], 
                               z=[left[2], arrow_end[2], right[2]], 
                               mode='lines', 
                               line=dict(color=color, width=5), showlegend=False))
    return fig

# =================== modules for computing the standard basis ===================
def compute_feasibility(basis,sigma,S):

    model = gp.Model()
    model.params.OutputFlag = 0
    model.params.FeasibilityTol = 1e-9
    model.params.OptimalityTol = 1e-9

    coeff = [model.addVar(vtype=GRB.INTEGER, lb=0, ub=100, name=f'coeff_{i}') for i in range(np.shape(S)[1])]

    # objective
    expr = gp.LinExpr()
    for i in range(len(coeff)):
        expr += coeff[i]
    model.setObjective(expr, GRB.MINIMIZE)

    # constraint
    for i in range(len(basis)):
        expr = gp.LinExpr()
        for j in range(len(coeff)):
            expr += S[i,j]*coeff[j]*sigma[j]
        model.addConstr(expr == basis[i], name=f'target{i}')
    
    model.update()
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return True    

def standard_basis(sigma, S, v, k,model='linear'):
    
    feasible_basis = []
    for val in [1,-1]:
        for i in range(3):
            basis = np.zeros(3)
            basis[i] = val
            if compute_feasibility(basis,sigma,S):
                feasible_basis.append(basis)
    
    return feasible_basis

def region_existence(sigma,k):
    lnk = [np.log(_) if _ > 0 else -np.inf for _ in k]

    propensity = mm.get_propensity()
    ChemName = ['X','Y','ATP','ADP']
    
    model = gp.Model()
    model.params.OutputFlag = 0 
    model.params.ScaleFlag = 0
    model.params.FuncNonlinear = 0
    model.params.FeasibilityTol = 1e-9
    model.params.OptimalityTol = 1e-9
    model.params.TimeLimit = 180
    model.params.NumericFocus = 3
    
    conc_lb = 1e-6
    x_ub = {'X':10.0, 'Y':300.0, 'ATP':1.0, 'ADP':1.0}
    lnx = {m:model.addVar(vtype=GRB.CONTINUOUS, lb=np.log(conc_lb), ub=np.log(x_ub[m]), name=f'lnx{m}') for m in ChemName}
    x = {m:model.addVar(vtype=GRB.CONTINUOUS, lb=conc_lb, ub=x_ub[m], name=f'x{m}') for m in ChemName}
    xi = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f'xi')
    
    # objective
    model.setObjective(xi, GRB.MINIMIZE)

    # concentration constraint
    
    for m in ['ATP','ADP']:
        model.addGenConstrExp(lnx[m],x[m],name=f'exp{m}',options=f"FuncPieces=-2 FuncPieceError=1e-5") # x = exp(lnx)
    model.addLConstr(x['ATP'] + x['ADP'] == 1.0, name=f'conservation')
    
    # flux direction constraint (propensity) : ここの制約は、null-reaction manifold上の点での制約。上の制約はx_{i-1}からx_iへ移動するときのフラックスの積分への制約
    for r in range(len(k)):
        
        if np.isinf(lnk[r]):
            continue

        expr = gp.LinExpr()
        for m in propensity[r].keys():
            expr += lnx[m]*propensity[r][m] 
        expr -= lnk[r] 

        if sigma[r] > 0:
            model.addLConstr(expr >= 0, name=f'propensity{r}')
        elif sigma[r] < 0:
            model.addLConstr(expr <= 0, name=f'propensity{r}')
        else:
            raise ValueError('invalid direction')

    model.update()
    model.optimize()
    
    return model.status == GRB.OPTIMAL

def region_transitivity(source,target):
    diff = np.abs(source - target)
    # 差が1以上の要素のインデックスを見つける
    indices = np.where(diff >= 1e-10)
    if len(indices) > 1:
        raise ValueError('invalid input')
    flip_rxn = indices[0]

    ChemName = ['X','Y','ATP','ADP']
    v, k = mm.get_param()[:2]
    S = mm.get_stoichiometry()
    N, R = np.shape(S)
    propensity = mm.get_propensity()
    lnk = [np.log(_) if _ > 0 else -np.inf for _ in k]

    model = gp.Model()
    model.params.OutputFlag = 0 
    model.params.ScaleFlag = 0
    model.params.FuncNonlinear = 0
    model.params.FeasibilityTol = 1e-9
    model.params.OptimalityTol = 1e-9
    model.params.TimeLimit = 180
    model.params.NumericFocus = 3
    
    conc_lb = 1e-6
    x_ub = {'X':10.0, 'Y':300.0, 'ATP':1.0, 'ADP':1.0}
    lnx = {i:{m:model.addVar(vtype=GRB.CONTINUOUS, lb=np.log(conc_lb), ub=np.log(x_ub[m]), name=f'lnx{i}_{m}') for m in ChemName} for i in ['source','target']}
    x = {i:{m:model.addVar(vtype=GRB.CONTINUOUS, lb=conc_lb, ub=x_ub[m], name=f'x{i}_{m}') for m in ChemName} for i in ['source','target']}
    v = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'v_rxn{j}') for j in range(R)]
    xi =  {m:model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f'slack_{m}_plus') for m in ChemName}
    eta =  {m:model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f'slack_{m}_minus') for m in ChemName}
    inf_norm = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f'inf_norm')
    
    # objective
    for m in ChemName:
        model.addLConstr(xi[m] <= inf_norm, name=f'inf_norm{m}_plus')
        model.addLConstr(eta[m] <= inf_norm, name=f'inf_norm{m}_minus')
    model.setObjective(inf_norm, GRB.MINIMIZE)

    # concentration constraint
    for i in x.keys():
        for m in ChemName:
            model.addGenConstrExp(lnx[i][m],x[i][m],name=f'exp{i}_{m}',options=f"FuncPieces=-2 FuncPieceError=1e-5") # x = exp(lnx)
   
   # conservation law of R + L
    for i in x.keys():
        model.addLConstr(x[i]['ATP'] + x[i]['ADP'] == 1.0, name=f'conservation_{i}')
    
    # Connection Constraints
    for n,m in enumerate(ChemName):
        if m == 'ADP':
            continue
        expr = gp.LinExpr()
        for r in range(R):
            expr += S[n,r]*v[r]
        model.addLConstr(expr + xi[m] - eta[m] == x['target'][m] - x['source'][m], name=f'connection{m}')

    # flux direction constraint (v)
    for r in range(R):
        if source[r] > 0:
            model.addLConstr(v[r] >= 0, name=f'flux{r}_plus')
        else:
            model.addLConstr(v[r] <= 0, name=f'flux{r}_minus')
    
    # flux direction constraint (propensity) : ここの制約は、null-reaction manifold上の点での制約。上の制約はx_{i-1}からx_iへ移動するときのフラックスの積分への制約
    margin = 1e-2
    for l in x.keys():
        for r in range(R):
            
            if np.isinf(lnk[r]):
                continue

            expr = gp.LinExpr()
            for m in propensity[r].keys():
                expr += lnx[l][m]*propensity[r][m] 
            expr -= lnk[r] 


            if r == flip_rxn and l == 'target':
                model.addLConstr(expr == 0, name=f'propensity{l}_{r}_flip')
            else:
                if source[r] > 0:
                    model.addLConstr(expr >= margin, name=f'propensity{l}_{r}')
                elif source[r] < 0:
                    model.addLConstr(expr <= -margin, name=f'propensity{l}_{r}')
                else:
                    raise ValueError('invalid direction')

    model.update()
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        objVal = model.objVal
    else:
        objVal = 1e10
    
    return objVal















