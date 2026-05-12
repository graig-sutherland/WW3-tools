import numpy as np
import matplotlib.pyplot as plt

def tanh_spacing(elev, hmin, hmax, hcenter, hbeta):
    T = 1.0 + np.tanh(hbeta*(1.0-hcenter))
    hscl = (0.5*T*hmax - hmin) / (0.5*T*hmin - hmin) # to make sure hmin is met at the coast
    vals = np.maximum(1, -elev)
    vals = (hmax-hscl*hmin)*0.5*(1 + np.tanh(hbeta*(vals-hcenter))) + hscl*hmin
    vals = np.maximum(vals, hmin)
    vals = np.minimum(vals, hmax)
    return vals

if __name__ == "__main__":
    h0 = 150
    hb = 0.01
    hmin, hmax = 4, 30
    elev = np.linspace(-4000, -1,1001)
    plt.figure()
    tspac = tanh_spacing(elev,hmin, hmax, h0, hb)
    plt.semilogx(-elev, tspac)
    plt.xlabel(r'depth / m')
    plt.ylabel('grid spacing / km')
    plt.savefig('tanh_spacing.png')
