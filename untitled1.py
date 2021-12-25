import os

import numpy as np
from numpy import polynomial as P 

from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.figsize": [8.0, 5.0]})

# D = 2.0 for the rest of the notebook
D = 2.0

# Energy values from 3b_solve_R.ipynb
mo_nums = {
    '1 sigma g': (1, 0, 0),
    '1 sigma u *': (1, 0, 1),
    '1 pi u': (1, 1, 1),
    '2 sigma g': (2, 0, 0),
    '2 sigma u *': (2, 0, 1),
    '3 sigma g': (1, 0, 2),
    '1 pi g *': (1, 1, 2),
    '3 sigma u *':(1, 0, 3),
}

mo_E_exact = {
    '1 sigma g': -1.0996287880084956167792142878170125186443328857421875,
    '1 sigma u *': -0.66747226771811252188371099691721610724925994873046875,
    '1 pi u': -0.428757630866889283272058719376218505203723907470703125,
    '2 sigma g': -0.360299648608283396011842114603496156632900238037109375,
    '2 sigma u *': -0.255397351472925360216237322674714960157871246337890625,
    '3 sigma g': -0.2360272886226555122402004371906514279544353485107421875,
    '1 pi g *': -0.2266972947658647818958144171119784004986286163330078125,
    '3 sigma u *': -0.1263147875605639403051583258275059051811695098876953125,
}

def get_lbda(l, mu):
    """Returns Lambda loaded from file"""
    filename = f"data_lbda/lbda_{mu}{l}.txt"
    filedir = os.path.join(os.path.abspath(''), filename)
    with open(filedir, 'r') as file:
        lbda_coef = np.loadtxt(file)
    lbda = P.Polynomial(lbda_coef) # a function of c2
    return lbda

def get_R(l, mu, E_exact):
    """Returns function R of a particular l, mu, and E_exact"""
    # greater values of l require greater cutoff values
    if l == 0:
        cutoff = 10
    elif l == 1:
        cutoff = 15
    elif l == 2:
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
    # evenness determined by the number of angular nodes between the two nuclei
    # excluding angular nodes that contain the internuclear axis
    is_even = True if (l - mu) % 2 == 0 else False
    
    c2 = (1/2) * E_exact * (2.0**2)
    lbda_val = get_lbda(l, mu)(c2)

    p1 = np.array([1, ((mu*(mu+1) + c2 - lbda_val)/(2*(mu+1)))])

    F = lambda eta, p: np.array([[0, 1], 
    [(mu*(mu+1) - lbda_val + c2*(eta**2))/(1 - eta**2), 2*(mu+1)*eta/(1 - eta**2)]]) @ p
    
    # obtain values of f(eta) along these points
    # we exclude eta=1 to avoid a ZeroDivisionError    
    eta_span = np.linspace(1, 0, 100)
    eps = 1.0e-10 # a very small value
    eta_span[0] = 1.0-eps

    # solve using solve_ivp, sol.y[0] is f(eta_span) and sol.y[1] is f'(eta_span)
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
    
    return interp1d(np.linspace(-1.0, 1.0, 199), S_vals, kind='quadratic')

# For mu = 0 only
"""
 Exact MO |  LCAO-MO
 sigma_u     sigma_u_1s 
 sigma_g     sigma_g_1s
"""
# Simpson's rule for fast integration
from scipy.integrate import simpson

# cross section of exact MO, along the xz-plane
# mu = 0, l = 0 (bonding) or 1 (antibonding)
def get_mo_cs(l):
    if l == 0:
        E_exact = mo_E_exact['1 sigma g']
    else:
        E_exact = mo_E_exact['1 sigma u *']
    R, cutoff = get_R(l, 0, E_exact)
    S = get_S(l, 0, E_exact)

    # R * S needs to be normalized
    # volume element = (xi**2 - eta**2) * d xi * d eta * d phi
    # integration with phi yields normalization constant of 1/sqrt(2*pi)
    
    # note: simpson requires even number of intervals (odd number of samples)
    eta_span = np.linspace(-1.0, 1.0, 101)
    S2_int = simpson(S(eta_span)**2, x=eta_span)
    eta2_S2_int = simpson((eta_span*S(eta_span))**2, x=eta_span)
    
    xi_span = np.linspace(1.0, cutoff, cutoff*50+1)
    R2_int = simpson(R(xi_span)**2, x=xi_span)
    xi2_R2_int = simpson((xi_span*R(xi_span))**2, x=xi_span)

    # norm_const = 1/A in the equation above
    norm_const = np.sqrt(2*np.pi*(S2_int*xi2_R2_int - eta2_S2_int*R2_int))
    
    # takes cartesian coordinates
    def mo_returned(x, z):
        # distances from nucleus A and B
        rA = np.sqrt(x**2 + (z + 1.0)**2)
        rB = np.sqrt(x**2 + (z - 1.0)**2)
        
        xi = (rA + rB)/2.0
        eta = (rA - rB)/2.0
        
        # check for bounds
        if xi < 1.0:
            xi = 1.0
        
        if eta < -1.0:
            eta = -1.0
        elif eta > 1.0:
            eta = 1.0
        
        # because we work with sigma orbitals only,
        # we do not deal with phi here

        # xi > cutoff -> R = 0 implemented here
        if xi > cutoff:
            return 0
        else:
            return (R(xi)*S(eta))/norm_const

    return mo_returned

# cross section of LCAO-MO, evaluated only on the xz-plane
# mu = 0, l = 0 (bonding) or 1 (antibonding)
def get_lcao_cs(l):
    # 1s orbital, function of r only
    s_orbital = lambda r: np.exp(-r)/np.sqrt(np.pi)
    
    # cartesian coordinates
    def lcao_returned(x, z, l):
        rA = np.sqrt(x**2 + (z + 1.0)**2)
        rB = np.sqrt(x**2 + (z - 1.0)**2)

        # normalization constants are from literature
        S_2 = np.exp(-2.0) * (1 + 2.0 + (2.0**2)/3)
        if l == 0: # bonding
            return (s_orbital(rB) + s_orbital(rA))/np.sqrt(2 * (1 + S_2))
        else: # antibonding
            return (s_orbital(rB) - s_orbital(rA))/np.sqrt(2 * (1 - S_2))

    return lambda x, z: lcao_returned(x, z, l)

# exact MOs
exact_mos = [
    np.vectorize(get_mo_cs(0)),
    np.vectorize(get_mo_cs(1)),
]

# LCAO-MOs
lcao_mos = [
    np.vectorize(get_lcao_cs(0)),
    np.vectorize(get_lcao_cs(1)),
]

z_span = np.linspace(-5, 5, 300)
# z_span = np.concatenate([
#     np.linspace(-5, 5, 300),
# ], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(18.0, 8.0))

for l in range(2):
    mo_z = np.vectorize(lambda z: exact_mos[l](0, z))
    axes[l].plot(z_span, mo_z(z_span))
    
    lcao_z = np.vectorize(lambda z: lcao_mos[l](0, z))
    axes[l].plot(z_span, lcao_z(z_span))
    
    axes[l].set_xlabel('z')
    axes[l].set_xlabel(r'$\psi$')

axes[0].set_title(r'Bonding Orbital $1 \sigma_g$')
axes[1].set_title('Antibonding Orbital $1 \sigma_u^*$')