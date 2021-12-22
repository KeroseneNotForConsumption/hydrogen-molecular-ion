import os
import numpy as np
from numpy import polynomial as P 
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('default')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.figsize": [8.0, 5.0]})

# D = 2, always
D = 2

# Energy values from 3b_solve_R.ipynb
E_exact_dict = {
    '1 sigma u *' : -0.6674722771701539,
    '1 sigma g'   : -1.0996287880084956}

def get_lbda(mu, l):
    # retrieve Lambda from file
    filename = f"data_lbda/lbda_{mu}{l}.txt"
    filedir = os.path.join(os.path.abspath(''), '..' , filename)
    with open(filedir, 'r') as file:
        lbda_coef = np.loadtxt(file)
    lbda = P.Polynomial(lbda_coef) # a function of c2
    return lbda

def get_R(mu, l, E_exact):
    
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
    lbda_val = get_lbda(mu, l)(c2)
    
    # initial value of g and g'
    q1 = np.array([1,-(2*D + c2 + mu*(mu+1) - lbda_val)/(2*(mu+1))])
    
    G = lambda xi, q: np.array([[0, 1],
    [-(2*D*xi + c2*(xi**2) + mu*(mu+1) - lbda_val)/(xi**2 - 1),
    -(2*(mu+1)*xi)/(xi**2 - 1)]]) @ q
    
    # solve for g(xi) for 1 < xi <= cutoff
    # exclude xi = 1 to avoid a divide-by-zero error
    xi_span_for_calc = np.linspace(1, cutoff, cutoff*100)
    eps = 1e-10
    xi_span_for_calc[0] = 1 + eps
    
    sol = solve_ivp(G, [1 + eps, cutoff], q1, t_eval=xi_span_for_calc)
    
    # calculate R from g
    R_vals = ((xi_span_for_calc ** 2 - 1)**(mu/2)) * sol.y[0]
    
    # add back xi = 0 and R(xi) = 0 or 1 depending on mu
    xi_span_for_calc[0] = 1
    R_vals[0] = 1 if mu == 0 else 0
    
    R_interp = interp1d(xi_span_for_calc, R_vals, kind='linear')
    
    # NOTE: xi greater than cutoff should result in R = 0
    # to be implemented with functions down the line
    return R_interp, cutoff

def get_S(mu, l, E_exact):
    
    # evenness determined by the number of angular nodes between the two nuclei
    # excluding angular nodes that contain the internuclear axis
    is_even = True if (l - mu) % 2 == 0 else False
    
    c2 = (1/2) * E_exact * (2.0**2)
    lbda_val = get_lbda(mu, l)(c2)

    p1 = np.array([1, ((mu*(mu+1) + c2 - lbda_val)/(2*(mu+1)))])

    F = lambda eta, p: np.array([[0, 1], 
    [(mu*(mu+1) - lbda_val + c2*(eta**2))/(1 - eta**2), 2*(mu+1)*eta/(1 - eta**2)]]) @ p
    
    # obtain values of f(eta) along these points (we exclude eta=1 to avoid a divide-by-zero error)
    eta_span = np.linspace(1, 0, 100)
    eps = 1e-10 # a very small value
    eta_span[0] = 1 - eps

    # solve using solve_ivp, sol.y[0] is f(eta_span) and sol.y[1] is f'(eta_span)
    sol = solve_ivp(F, [1 - eps, 0], p1, t_eval=eta_span)

    # we are plotting S(eta), not f(eta)
    S_vals = ((1 - eta_span**2) ** (mu/2)) * sol.y[0]

    # add back S_val(1) to eta_span
    S_vals[0] == 1 if mu == 0 else 0
    

    if is_even:
        S_vals = np.concatenate([S_vals[:-1], S_vals[::-1]])
    else:
        S_vals = np.concatenate([-S_vals[:-1], S_vals[::-1]])
    
    return interp1d(np.linspace(-1.0, 1.0, 199), S_vals, kind='linear')


# note: D = 2.0
def cart_to_ellip(x, y, z):
    # define point A and B
    cart_coords = np.array([x, y, z])
    pA = np.array([0, 0, -1.0])
    pB = np.array([0, 0, 1.0])
    rA = np.linalg.norm(cart_coords - pA)
    rB = np.linalg.norm(cart_coords - pB)
    
    xi = (rA + rB)/2.0
    nu = (rA - rB)/2.0
    
    # check for bounds
    if xi < 1.0:
        xi = 1.0
    
    if nu < -1.0:
        nu = -1.0
    elif nu > 1.0:
        nu = 1.0
    
    # getting phi to work is tricky
    if cart_coords[0] == 0.0:
        phi = np.pi/2 if cart_coords[1] >= 0 else (3*np.pi)/2
    else:
        phi = np.arctan(cart_coords[1] / cart_coords[0])
        if cart_coords[0] <= 0:
            phi += np.pi
    
    return xi, nu, phi
    

# n, mu, l of mos
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

E_exact_dict = {
    '1 sigma g': -1.0996287880018977833884719075285829603672027587890625,
    '1 sigma u *': -0.66747226769876011331916743074543774127960205078125,
    '1 pi u': -0.428757630933387978711124333131010644137859344482421875,
    '2 sigma g': -0.360299650543231930743814928064239211380481719970703125,
    '2 sigma u *': -0.255397764046394915027349270530976355075836181640625,
    '3 sigma g': -0.2360276423296774617188731326677952893078327178955078125,
    '1 pi g *': -0.2266979721961866200974355933794868178665637969970703125,
    '3 sigma u *':-0.12664478743642726410456589292152784764766693115234375,
}
# returns the (xi, eta) coordinates of the chosen samples

# random number generator from numpy
rng = np.random.default_rng()

# change this into a loop
for mo_notation in mo_nums.keys():
    n, mu, l = mo_nums[mo_notation]
    E_exact = E_exact_dict[mo_notation]
    
    R, cutoff = get_R(mu, l, E_exact)
    S = get_S(mu, l, E_exact)
    
    """
    xi_span_org = np.linspace(1.0, cutoff, cutoff*20)
    scaled_xi = ((xi_span_org - 1.0)/(cutoff - 1.0))**1.5
    xi_span = scaled_xi * (cutoff - 1.0) + 1.0
    """
    
    d_xi = (cutoff - 1.0)/(cutoff*100)
    xi_span = np.linspace(1.0+(d_xi/2), cutoff-(d_xi/2), cutoff*100 - 1)
    xi_size = xi_span.size
    d_xi = (xi_span[1] - xi_span[0])
    
    d_eta = 2.0/(50 * (l+1))
    eta_span = np.linspace(-1.0+(d_eta/2), 1.0-(d_eta/2), 50*(l+1) - 1)
    eta_size = eta_span.size
    d_eta = (eta_span[1] - eta_span[0])
    
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
        PHI_sign = np.full(sample_size, True)
        
        points_sign = RS_sign
    else:
        # PHI  - cos(mu*phi) only
        d_phi = 2*np.pi/50
        phi_span = np.linspace(0, 2*np.pi - d_phi, 50)
        
        PHI_values = np.cos(mu * phi_span)
        
        PHI_prob_values = (PHI_values ** 2).flatten()
        PHI_prob_values /= np.sum(PHI_prob_values)
        
        # sign of samples
        PHI_sign = PHI_values > 0.0
        
        PHI_samples = rng.choice(50, sample_size, p=PHI_prob_values)
        # sign of samples
        PHI_sign = PHI_sign[PHI_samples]
        
        points_sign = np.logical_not(np.logical_xor(RS_sign, PHI_sign))
        
        PHI_points = np.reshape(phi_span[PHI_samples], newshape=(1, sample_size))
    
    
    points = np.concatenate((RS_points, PHI_points), axis=0)
    points = points.T
    # add randomness
    
    
    """
    points += np.concatenate((rng.uniform(-d_xi/2, +d_xi/2, size=(1, sample_size)),
                          rng.uniform(-d_eta/2, +d_eta/2, size=(1, sample_size)),
                          rng.uniform(0, 2*np.pi, size=(1, sample_size))))
    """
    
    def ellip_to_cart(ellip_coords):
        xi, eta, phi = ellip_coords
        r2 = (xi**2 - 1.0) * (1.0 - eta**2)
        if r2 > 0:
            r = np.sqrt(r2)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
        else:
            print(f' {xi**2, eta**2} ')
            x = y = 0
        return np.array([x, y, (xi*eta)])
    ellip_to_cart = np.vectorize(ellip_to_cart, signature='(3)->(3)')
    
    points_cart = ellip_to_cart(points)
    
    points_sign = np.reshape(points_sign, newshape=(sample_size, 1))
    samples_to_be_saved = np.concatenate((points_cart, points_sign), axis=1)
    
    np.save(f'samples_{n}{mu}{l}.npy', samples_to_be_saved)

