def carreau_yasuda_viscosity(gamma_dot, eta0=0.056, eta_inf=0.0035, lam=3.313, n=0.3568, a=2.0):
    """
    Carreau-Yasuda model for blood viscosity.
    
    eta0    = 0.056  Pa·s  — thick blood at rest (RBCs clumped)
    eta_inf = 0.0035 Pa·s  — thin blood at high speed (RBCs dispersed)
    lam     = 3.313  s     — time constant: where thinning kicks in
    n       = 0.3568       — how steep the thinning is (< 1 = shear-thinning)
    a       = 2.0          — shape of the thinning curve
    
    gamma_dot: shear rate tensor |du/dy|, shape [N]
    returns: viscosity tensor, same shape [N]
    """
    return eta_inf + (eta0 - eta_inf) * (1.0 + (lam * gamma_dot) ** a) ** ((n - 1.0) / a)