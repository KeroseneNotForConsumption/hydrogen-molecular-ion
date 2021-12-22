import os
import numpy as np
from numpy import polynomial as P 
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

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
    rA = np.linalg.norm(np.array([x, y, z-1.0]))
    rB = np.linalg.norm(np.array([x, y, z+1.0]))
    
    xi = (rA + rB)/2.0
    nu = (rA - rB)/2.0
    
    # check for bounds
    if xi < 1.0:
        xi = 1.0
    
    if nu < -1.0:
        nu = -1.0
    elif nu > 1.0:
        nu = 1.0
    
    # output of phi: from -pi to +pi
    phi = np.arctan2(y, x)
    
    return xi, nu, phi
    

# n, mu, l of mos


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
from scipy.integrate import simpson
from skimage import measure


# random number generator from numpy
rng = np.random.default_rng()

# change this into a loop
mo_isoval = {
    '1 sigma g': 0.055,
    '1 sigma u *': 0.038,
    '1 pi u': 0.022,
    '2 sigma g': 0.014,
    '2 sigma u *': 0.0104,
    '3 sigma g': 0.011,
    '1 pi g *': 0.011,
    '3 sigma u *': 0.0051,
}


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

mo_range = {
    '1 sigma g': 2,
    '1 sigma u *': 3,
    '1 pi u': 6,
    '2 sigma g': 7,
    '2 sigma u *': 8,
    '3 sigma g': 9,
    '1 pi g *': 9,
    '3 sigma u *': 15,
}

for mo_notation in mo_nums.keys():
    
    n, mu, l = mo_nums[mo_notation]
    E_exact = E_exact_dict[mo_notation]
    
    R, cutoff = get_R(mu, l, E_exact)
    S = get_S(mu, l, E_exact)
    
    eta_span = np.linspace(-1.0, 1.0, 50*(l-mu+1)+1)
    S2_int = simpson(S(eta_span)**2, x=eta_span)
    eta2_S2_int = simpson((eta_span*S(eta_span))**2, x=eta_span)
    
    xi_span = np.linspace(1.0, cutoff, cutoff*50)
    R2_int = simpson(R(xi_span)**2, x=xi_span)
    xi2_R2_int = simpson((xi_span*R(xi_span))**2, x=xi_span)
    
    
    norm_const_RS = np.sqrt(S2_int*xi2_R2_int - eta2_S2_int*R2_int)
    
    if mu != 0:
        norm_const_PHI = np.sqrt(np.pi)
    
    @np.vectorize
    def mo_func(x, y, z):
        xi, eta, phi = cart_to_ellip(x, y, z)
        if xi > cutoff:
            return 0.0
        else:
            if mu == 0:
                return R(xi)*S(eta) / (norm_const_RS * np.sqrt(2*np.pi))
            else:
                return R(xi)*S(eta)*np.cos(mu*phi) / (norm_const_RS * np.sqrt(np.pi))
    
    
    
    delta = mo_range[mo_notation]/25.0
    z_range = mo_range[mo_notation] * 2.5
    xy_range = z_range * 0.7
    
    xy_span = np.linspace(-xy_range, xy_range, int((2*xy_range)/delta))
    z_span = np.linspace(-z_range, z_range, int((2*z_range)/delta))
    X, Y, Z = np.meshgrid(xy_span, xy_span, z_span, 
                          indexing='ij', sparse=True)
    
    del_xy = xy_span[1]-xy_span[0]
    del_z = z_span[1]-z_span[0]
    
    @np.vectorize
    def xy_convert(ind):
        return ind - xy_range
     
    @np.vectorize
    def z_convert(ind):
        return ind - z_range   
    
    vol = mo_func(X, Y, Z)
    
    
    iso_val = mo_isoval[mo_notation]
    
    
    
    verts1, faces1, _, _ = measure.marching_cubes(vol, level=iso_val,
                                        spacing=(del_xy, del_xy, del_z))
    verts1_shifted = np.vstack([
        verts1[:, 0] - xy_range, 
        verts1[:, 1] - xy_range,
        verts1[:, 2] - z_range,]).T
    
    np.save(f'isosurface_{n}{mu}{l}_verts_pos.npy', verts1_shifted)
    np.save(f'isosurface_{n}{mu}{l}_faces_pos.npy', faces1)
    
    
    if n + mu + l > 1: 
        verts2, faces2, _, _ = measure.marching_cubes(vol, level=-iso_val,
                                            spacing=(del_xy, del_xy, del_z))
        verts2_shifted = np.vstack([
            verts2[:, 0] - xy_range, 
            verts2[:, 1] - xy_range,
            verts2[:, 2] - z_range,]).T
        
        np.save(f'isosurface_{n}{mu}{l}_verts_neg.npy', verts2_shifted)
        np.save(f'isosurface_{n}{mu}{l}_faces_neg.npy', faces2)
    
    
        
        
