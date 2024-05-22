import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def ConstructMatrix(z, sigma, k, S, subs, prod, model='cell'):
    # variables are assigned as === x ===, === v ===, === lam ===
    N, R = np.shape(S)
    V = np.shape(z)[0]
    if V == 1:
        z = np.vstack((z,z))
        V = 2
    xmax = [10 for i in range(N)]
    if model == 'Brusselator':
        xmax = [2,4]
    vmax = [20 for i in range(R)]
    
    # set up index dictionary for convenience
    idx = {}
    for i in range(R):
        idx[f'v{i}'] = i
    for i in range(V-1):
        idx[f'lam{i}'] = i + R
    
    # set up matrix A and vector b
    A = np.zeros((2*N+3*R+V,R+V-1))
    b = np.zeros((2*N+3*R+V,))

    # concentration upper bound
    for n in range(N):
        for v in range(V-1):
            A[n,idx[f'lam{v}']] = z[v,n] - z[V-1,n]
        for r in range(R):
            A[n,idx[f'v{r}']] = - S[n,r]
        b[n] = xmax[n] - z[V-1,n]
    
    # concentration lower bound
    for n in range(N):
        for v in range(V-1):
            A[N+n,idx[f'lam{v}']] = - (z[v,n] - z[V-1,n])
        for r in range(R):
            A[N+n,idx[f'v{r}']] = S[n,r]
        
        b[N+n] = z[V-1,n]

    # flux direction consistency
    for r in range(R):
        p = prod[r][0]

        if subs[r] == []:
            for v in range(V-1):
                A[2*N+r,idx[f'lam{v}']] = sigma[r]*k[r]*(z[v,p] - z[V-1,p])
            for i in range(R):
                A[2*N+r,idx[f'v{i}']] = - sigma[r]*k[r]*S[p,i]
        
            b[2*N+r] = sigma[r]*(1 - k[r]*z[V-1,p])
        else:
            s = subs[r][0]
            for v in range(V-1):
                A[2*N+r,idx[f'lam{v}']]  = - sigma[r]*(z[v,s] - z[V-1,s])
                A[2*N+r,idx[f'lam{v}']] += sigma[r]*k[r]*(z[v,p] - z[V-1,p])
            
            for i in range(R):
                A[2*N+r,idx[f'v{i}']] = sigma[r]*S[s,i]
                A[2*N+r,idx[f'v{i}']] += - sigma[r]*k[r]*S[p,i]
        
            b[2*N+r] = sigma[r]*(z[V-1,s] - k[r]*z[V-1,p])
        
    # flux direction consistency 2 
    for r in range(R):
        A[2*N+R+r,idx[f'v{r}']] = -sigma[r]
    
    # flux upper bound
    for r in range(R):
        A[2*N+2*R+r,idx[f'v{r}']] = sigma[r]
        b[2*N+2*R+r] = vmax[r]
    
    # lambda lower bound
    for i in range(V-1):
        A[2*N+3*R+i,idx[f'lam{i}']] = -1
    
    # lambda sum
    for i in range(V-1):
        A[2*N+3*R+V-1,idx[f'lam{i}']] = 1
        b[2*N+3*R+V-1] = 1
    
    # round values to help the solver
    # use original value if the element is too small
     
    A[abs(A) <= 1e-10] = 0
    b[abs(b) <= 1e-10] = 0

    return A, b

def compute_flux(x,k):
    p = [0,0,0]
    p[0] = 1.0 - k[0]*x[0]
    p[1] = x[0] - k[1]*x[1]
    p[2] = x[1] - k[2]*x[0]
    return p

def compute_flux_sign(x,k):
    p = [0,0,0]
    p[0] = 1.0 - k[0]*x[0]
    p[1] = x[0] - k[1]*x[1]
    p[2] = x[1] - k[2]*x[0]
    return [int(np.sign(p[i])) for i in range(3)]

def reduce_vertices(vertices,S,z):
    N, R = np.shape(S)
    V = np.shape(z)[0]
    flux = [v[:R] for v in vertices]
    lam = [np.array(list(v[R:])+[1-sum(v[R:])]) for v in vertices]
    #print('flux')
    #print(flux)
    #print('lam')
    #print(lam)

    tmp = []
    p = len(vertices)
    for i in range(p):
        x = [0 for n in range(N)]
        for n in range(N):
            for v in range(V):
                x[n] += z[v,n]*lam[i][v]
            for r in range(R):
                x[n] -= S[n,r]*flux[i][r]
        tmp.append(x)
        
    #vertices = [np.dot(z,l) - np.dot(S,f) for l,f in zip(lam,flux)]
    #print('tmp vs vertices')
    #print(vertices)
    #print(tmp)
    #print()
    vertices = tmp[:]
    #print(f'computed vertices = {vertices}')
    CUTOFF = 1e-4
    new_vertices = []
    for v in vertices:
        if any([np.linalg.norm(np.array(v)-np.array(u)) < CUTOFF for u in new_vertices]):
            pass
        else:
            new_vertices.append(v)
    return np.array(new_vertices)

def vertices_on_manifolds_Brusselator(vertices,k):
    threshold = 1e-4
    # calculate the flux on the vertices
    flux = [compute_flux(v,k) for v in vertices]
    #print(flux)
    # list up the vertices on the manifolds
    vertices_on_manifolds = [np.array([v for v,f in zip(vertices,flux) if abs(f[i]) < threshold]) for i in range(3)]
    return vertices_on_manifolds


def sort_points(points):
    # Sort the points to form a continuous boundary of the polygon
    # This is a simple approach and may need adjustments for complex shapes
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def plot_initialize(k,Polygon=None):
    # Set the font globally
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Times'
    mpl.rcParams['font.size'] = 20


    # plot the line 
    x_line = np.linspace(0,2,100)
    p0 = [[1/k[0],y] for y in np.linspace(-1,4,100)]
    p1 = [[x,x/k[1]] for x in np.linspace(-1,2,100)]
    p2 = [[x,x*k[2]] for x in np.linspace(-1,2,100)]
    fig, ax = plt.subplots(figsize=(8, 6))
    #ax = plt.axes(aspect='equal')
    plt.plot([x[0] for x in p0],[x[1] for x in p0],"-",color='k')
    plt.plot([x[0] for x in p1],[x[1] for x in p1],"--",color='k')
    plt.plot([x[0] for x in p2],[x[1] for x in p2],"-.",color='k')
    # add legend
    #plt.legend(['p0','p1','p2'],loc='upper left')

    # set xrange and yrange from 0 to 4
    plt.xlim([0.0,2.0])
    plt.ylim([0.0,4.])
    #set xtics
    plt.xticks([0.0,0.5,1,1.5,2.0])
    plt.yticks([0.0,1.0,2.0,3.0,4.0])
    # 小数点1桁まで表示
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    
    return fig, ax

def plot_polytope(fig,ax,start_polytope_vetices,vertices,color_idx,hatch='',alpha=0.5):
    color_options = ['#87CEEB',  # Sky Blue
                 '#FF7F50',  # Coral
                 '#32CD32',  # Lime Green
                 '#F4A460',  # Sandy Brown
                    '#ffffff']  # White
    sorted_points = sort_points(vertices)

    # Create a polygon
    polygon = plt.Polygon(sorted_points, closed=True, fill=True, hatch=hatch,color=color_options[color_idx], alpha=alpha)

    # Plotting
    ax.add_patch(polygon)
    
    for i in range(np.shape(start_polytope_vetices)[0]):
        plt.scatter(start_polytope_vetices[i,0],start_polytope_vetices[i,1],marker='*',color='#666666',s=120)
    



