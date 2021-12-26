import os

import numpy as np
from numpy import polynomial as P 

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


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

# random number generator from numpy
rng = np.random.default_rng()

for mo in mos:
    n, l, mu = mo_nums[mo]
    E_exact = mo_E_exact[mo]
    
    # we first deal with xi and eta, phi is dealt later
    R, cutoff = get_R(l, mu, E_exact)
    S = get_S(l, mu, E_exact)
    
    # the minimum and maximum values of xi and eta are purposefully avoided 
    d_xi = (cutoff-1.0)/(cutoff*100)
    xi_span = np.linspace(1.0+(d_xi/2), cutoff-(d_xi/2), cutoff*100-1)
    xi_size = xi_span.size
    d_xi = (xi_span[1]-xi_span[0])
    
    d_eta = 2.0/(50 * (l+1))
    eta_span = np.linspace(-1.0+(d_eta/2), 1.0-(d_eta/2), 50*(l+1)-1)
    eta_size = eta_span.size
    d_eta = (eta_span[1]-eta_span[0])
    
    XI, ETA = np.meshgrid(xi_span, eta_span, indexing='ij', sparse=True)
    
    @np.vectorize
    def RS(xi, eta):
        return R(xi) * S(eta)
    
    RS_values = RS(XI, ETA)
    # True = positive, False = negative
    RS_sign = RS_values > 0.0
    
    # normalized probability mass function
    RS_prob_values = ((RS_values ** 2) * (XI**2 - ETA**2)).flatten()
    RS_prob_values /= np.sum(RS_prob_values)
    
    # sampled indices
    sample_size = 2500
    RS_samples = rng.choice(xi_size*eta_size, 
                     sample_size, p=RS_prob_values)
    # sign of samples
    RS_sign = RS_sign.flat[RS_samples]
    
    # into indices for xi and eta
    RS_samples = np.unravel_index(RS_samples, (xi_size, eta_size))
    
    # organizing into points
    RS_points = np.vstack((xi_span[RS_samples[0]], eta_span[RS_samples[1]]))
    
    if mu == 0: # for sigma orbitals
        PHI_points = rng.uniform(0, 2*np.pi, size=(1, sample_size))
        PHI_sign = np.full(sample_size, True) # All values positive
        
        points_sign = RS_sign
    else: # PHI(phi) = cos(mu*phi) only
        d_phi = 2*np.pi/50
        phi_span = np.linspace(0, 2*np.pi - d_phi, 50)
        
        PHI_values = np.cos(mu * phi_span)
        
        PHI_prob_values = (PHI_values ** 2).flatten()
        PHI_prob_values /= np.sum(PHI_prob_values)
        
        # sign of PHI at all phi
        PHI_sign = PHI_values > 0.0
        
        PHI_samples = rng.choice(50, sample_size, p=PHI_prob_values)
        # sign of PHI of the samples
        PHI_sign = PHI_sign[PHI_samples]
        
        points_sign = np.logical_not(np.logical_xor(RS_sign, PHI_sign))
        
        PHI_points = np.reshape(phi_span[PHI_samples], newshape=(1, sample_size))
    
    
    points = np.concatenate((RS_points, PHI_points), axis=0)
    points = points.T
    
    def ellip_to_cart(ellip_coords):
        xi, eta, phi = ellip_coords
        r2 = (xi**2 - 1.0) * (1.0 - eta**2)
        if r2 > 0:
            r = np.sqrt(r2)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
        else: # must be on the internuclear axis
            x = y = 0
        return np.array([x, y, (xi*eta)])
    ellip_to_cart = np.vectorize(ellip_to_cart, signature='(3)->(3)')
    
    points_cart = ellip_to_cart(points)
    
    points_sign = np.reshape(points_sign, newshape=(sample_size, 1))
    array_to_be_saved = np.concatenate((points_cart, points_sign), axis=1)
    
    np.save(f'samples_{n}{l}{mu}.npy', array_to_be_saved)

