# this script was used to calculate 
# the appropriate value for the isosurfaces (mo_isoval)

import os

import numpy as np
from numpy import polynomial as P 

from scipy.integrate import solve_ivp, simpson
from scipy.interpolate import interp1d

from skimage import measure

import matplotlib.pyplot as plt

# D = 2.0, always
D = 2.0

def get_lbda(l, mu):
    """Returns Lambda loaded from file"""
    filename = f"lbda_{l}{mu}.txt"
    filedir = os.path.join(os.path.abspath(''), 
                           '..',
                           'lbda',
                           filename)
    with open(filedir, 'r') as file:
        lbda_coef = np.loadtxt(file)
    lbda = P.Polynomial(lbda_coef) # A function of c2
    return lbda

def get_R(l, mu, E_exact):
    """Returns function R of a particular l, mu, and E_exact"""
    # greater values of l require greater cutoff values
    if l == 0:
        cutoff = 15
    elif l == 1 or l == 2:
        cutoff = 20
    elif l == 3:
        cutoff = 40
    
    c2 = (1/2) * E_exact * (D**2)
    lbda_val = get_lbda(l, mu)(c2)
    
    # initial value of g and g' (at xi = 1)
    q1 = np.array([1,-(2*D + c2 + mu*(mu+1) - lbda_val)/(2*(mu+1))])
    
    # relates g' and g'' with g and g'
    G = lambda xi, q: np.array([[0, 1],
    [-(2*D*xi + c2*(xi**2) + mu*(mu+1) - lbda_val)/(xi**2 - 1),
    -(2*(mu+1)*xi)/(xi**2 - 1)]]) @ q
    
    # solve for g(xi) for 1 < xi <= cutoff
    # exclude xi = 1 to avoid a ZeroDivisionError
    xi_span = np.linspace(1, cutoff, cutoff*100)
    eps = 1e-10
    xi_span[0] = 1.0+eps
    
    sol = solve_ivp(G, [1.0 + eps, cutoff], q1, t_eval=xi_span)
    
    # calculate R from g
    R_vals = ((xi_span ** 2 - 1)**(mu/2)) * sol.y[0]
    
    # add back xi = 0 and R(xi) = 0 or 1 depending on mu
    xi_span[0] = 1.0
    R_vals[0] = 1.0 if mu == 0 else 0.0
    
    R_interp = interp1d(xi_span, R_vals, kind='quadratic')
    
    # NOTE: xi greater than cutoff should result in R = 0
    # to be implemented with functions down the line
    return R_interp, cutoff

def get_S(l, mu, E_exact):
    """Returns function S of a particular l, mu, and E_exact"""
    # evenness determined by 
    # the number of angular nodes between the two nuclei
    # excluding angular nodes that contain the internuclear axis
    is_even = True if (l - mu) % 2 == 0 else False
    
    c2 = (1/2) * E_exact * (2.0**2)
    lbda_val = get_lbda(l, mu)(c2)

    p1 = np.array([1, ((mu*(mu+1) + c2 - lbda_val)/(2*(mu+1)))])

    F = lambda eta, p: np.array([[0, 1], 
    [(mu*(mu+1) - lbda_val + c2*(eta**2))
     /(1 - eta**2), 2*(mu+1)*eta/(1 - eta**2)]]) @ p
    
    # obtain values of f(eta) along these points
    # we exclude eta=1 to avoid a ZeroDivisionError    
    eta_span = np.linspace(1, 0, 100)
    eps = 1.0e-10 # a very small value
    eta_span[0] = 1.0-eps

    # solve using solve_ivp, sol.y[0] is f(eta_span) 
    # and sol.y[1] is f'(eta_span)
    sol = solve_ivp(F, [1 - eps, 0], p1, t_eval=eta_span)

    # we are plotting S(eta), not f(eta)
    S_vals = ((1 - eta_span**2) ** (mu/2)) * sol.y[0]

    # add back S_val(1) to eta_span
    S_vals[0] == 1 if mu == 0 else 0
    
    # using symmetry to construct S(eta) for -1 <= eta < 0
    if is_even:
        S_vals = np.concatenate([S_vals[:-1], S_vals[::-1]])
    else:
        S_vals = np.concatenate([-S_vals[:-1], S_vals[::-1]])
    
    return interp1d(np.linspace(-1.0, 1.0, 199), 
                    S_vals, kind='quadratic')


mos = [
    '1 sigma g',
    '1 sigma u *',
    '1 pi u',
    '2 sigma g',
    '2 sigma u *',
    '3 sigma g',
    '1 pi g *',
    '3 sigma u *',
]

# n, l, mu
# same as the n, l, abs(m) of the AO of the united atom limit
mo_nums = {
    '1 sigma g': (1, 0, 0),
    '1 sigma u *': (2, 1, 0),
    '1 pi u': (2, 1, 1),
    '2 sigma g': (2, 0, 0),
    '2 sigma u *': (2, 1, 0),
    '3 sigma g': (3, 2, 0),
    '1 pi g *': (3, 2, 1),
    '3 sigma u *':(4, 3, 0),
}

mo_E_exact = {
    '1 sigma g': -1.09962878800462249273550696671009063720703125,
    '1 sigma u *': -0.66747226770434242570928518034634180366992950439453125,
    '1 pi u': -0.42875763089486185197785061973263509571552276611328125,
    '2 sigma g': -0.360299650542958704857454677039640955626964569091796875,
    '2 sigma u *': -0.25539776405624359245649657168542034924030303955078125,
    '3 sigma g': -0.236027642330498721445763976589660160243511199951171875,
    '1 pi g *': -0.2266979722153245335736215793076553381979465484619140625,
    '3 sigma u *': -0.126644787436385575229991218293434940278530120849609375,
}

# isosurface value, contains approx 90% of psi^2
# determined one by one
mo_isoval = {
    '1 sigma g': 0.0547,
    '1 sigma u *': 0.0383,
    '1 pi u': 0.0217,
    '2 sigma g': 0.0139,
    '2 sigma u *': 0.0104,
    '3 sigma g': 0.0112,
    '1 pi g *': 0.0111,
    '3 sigma u *': 0.00515,
}

mo_range = {
    '1 sigma g': 2.5,
    '1 sigma u *': 3,
    '1 pi u': 6,
    '2 sigma g': 7,
    '2 sigma u *': 8,
    '3 sigma g': 9,
    '1 pi g *': 9,
    '3 sigma u *': 15,
}

# note: D = 2.0
def cart_to_ellip(x, y, z):
    # define point A and B
    rA = np.sqrt(x**2 + y**2 + (z + 1.0)**2)
    rB = np.sqrt(x**2 + y**2 + (z - 1.0)**2)
    
    xi = (rA + rB)/2.0
    nu = (rA - rB)/2.0
    
    # check for bounds
    if xi < 1.0:
        xi = 1.0
    
    if nu < -1.0:
        nu = -1.0
    elif nu > 1.0:
        nu = 1.0
    
    return xi, nu, np.arctan2(y, x)+np.pi


for mo in mos:
    
    n, l, mu = mo_nums[mo]
    E_exact = mo_E_exact[mo]
    
    R, cutoff = get_R(l, mu, E_exact)
    S = get_S(l, mu, E_exact)
    
    
    
    
    # normalization for xi and eta
    eta_span = np.linspace(-1.0, 1.0, 50*(l-mu+1)+1)
    S2_int = simpson(S(eta_span)**2, x=eta_span)
    eta2_S2_int = simpson((eta_span*S(eta_span))**2, x=eta_span)
    
    xi_span = np.linspace(1.0, cutoff, cutoff*50+1)
    R2_int = simpson(R(xi_span)**2, x=xi_span)
    xi2_R2_int = simpson((xi_span*R(xi_span))**2, x=xi_span)
    
    norm_const_RS = np.sqrt(S2_int*xi2_R2_int - eta2_S2_int*R2_int)
    
    fig, ax = plt.subplots()
    ax.plot(np.linspace(1.0, cutoff, 100), R(np.linspace(1.0, cutoff, 100)))
    plt.show()
    
    # the wavefunction
    @np.vectorize
    def mo_func(x, y, z):
        xi, eta, phi = cart_to_ellip(x, y, z)
        if xi > cutoff:
            return 0.0
        else:
            if mu == 0:     # sigma orbitals
                return R(xi)*S(eta) / (norm_const_RS * np.sqrt(2*np.pi))
            else:           # pi, delta, etc.
                return R(xi)*S(eta)*np.cos(mu*phi) / (norm_const_RS * 
                                                      np.sqrt(np.pi))
    
    delta = mo_range[mo]/25.0
    z_range = mo_range[mo] * 2.5
    xy_range = z_range * 0.7
    
    xy_span = np.linspace(-xy_range, xy_range, int((2*xy_range)/delta))
    z_span = np.linspace(-z_range, z_range, int((2*z_range)/delta))
    X, Y, Z = np.meshgrid(xy_span, xy_span, z_span, 
                          indexing='ij', sparse=True)
    
    d_xy = xy_span[1]-xy_span[0]
    d_z = z_span[1]-z_span[0]
    
    @np.vectorize
    def xy_convert(ind):
        return ind - xy_range
     
    @np.vectorize
    def z_convert(ind):
        return ind - z_range   
    
    # the wavefunction, finally evaluated at regularly spaced points
    psi = mo_func(X, Y, Z)
    
    norm = ((psi**2).sum()) * (d_xy**2) * d_z
    print(f'psi^2 integrated, should equal 1: {norm}')
    
    
    def percentage_covered(iso_val):
    
        pos_values = psi[psi > iso_val]
        neg_values = psi[psi < -iso_val]
        
        result = (pos_values**2).sum() + (neg_values**2).sum()
        result *= (d_xy**2) * d_z
        
        print(f'Coverage percentage: {(result/norm)*100:.4f}%')
    
    percentage_covered(mo_isoval[mo])
    
    def plot_isosurf(iso_val):
        fig, ax = plt.subplots(subplot_kw={"projection":'3d'})
        
        # positive isosurface (red surface)
        verts_pos, faces_pos, _, _ = \
            measure.marching_cubes(psi, 
                                   level=iso_val,
                                   spacing=(d_xy, d_xy, d_z))
            
        verts_pos_shifted = np.vstack([
            verts_pos[:, 0] - xy_range, 
            verts_pos[:, 1] - xy_range,
            verts_pos[:, 2] - z_range,]).T
        
        
        ax.plot_trisurf(verts_pos_shifted[:, 0], 
                        verts_pos_shifted[:, 1], 
                        faces_pos, 
                        verts_pos_shifted[:, 2],
                        color='red', lw=1, alpha=0.6)
        
        # negative isosurface (blue surface) - excludes 1 sigma g
        if n + mu + l > 1: 
            verts_neg, faces_neg, _, _ = \
                measure.marching_cubes(psi, 
                                       level=-iso_val,
                                       spacing=(d_xy, d_xy, d_z))
                
            verts_neg_shifted = np.vstack([
                verts_neg[:, 0] - xy_range, 
                verts_neg[:, 1] - xy_range,
                verts_neg[:, 2] - z_range,]).T
            
            
            ax.plot_trisurf(verts_neg_shifted[:, 0], 
                            verts_neg_shifted[:, 1], 
                            faces_neg, 
                            verts_neg_shifted[:, 2],
                            color='blue', lw=1, alpha=0.6)
        
        ax.set_box_aspect((d_xy, d_xy, d_z))
        plt.show()
    
    #plot_isosurf(mo_isoval[mo])
    
