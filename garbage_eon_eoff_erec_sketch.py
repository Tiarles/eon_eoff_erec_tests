# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:06:46 2019

@author: tiarl
"""

import matplotlib.pyplot as plt
import numpy as np

# MODELING A POLY FROM ABB Documentation
# 5SYA2053-03ApplyingIGBTs.pdf

Ic_pts = np.array([100, 500, 1000, 1500, 2000, 2400])
Ic_pts = Ic_pts.reshape(Ic_pts.size, 1)

Ic_pts_ext = np.linspace(0, 2500, 2500)
Ic_pts_ext = Ic_pts_ext.reshape(Ic_pts_ext.size, 1)

E_on_pts = np.array([.5, .75, 1.5, 2.5, 3.75, 4.875])
E_off_pts = np.array([.25, .875, 1.75, 2.375, 3.0, 3.375])

from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 

poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(Ic_pts)

poly.fit(X_poly, E_on_pts)

lin2 = LinearRegression()
lin2.fit(X_poly, E_on_pts) 

 
plt.scatter(Ic_pts, E_on_pts, color = 'blue') 
plt.plot(Ic_pts_ext, lin2.predict(poly.fit_transform(Ic_pts_ext)), color = 'red') 
plt.title('Polynomial Regression (Prediction Test)') 
plt.xlabel('$I_c$ [A]') 
plt.ylabel('$E_{on}$ [J]')
plt.show() 

print(lin2.coef_)

#%%

Ic = Ic_pts_ext
Eon = .4 + 5.63340539e-04*Ic + 5.48091255e-07*Ic**2

plt.plot(Ic_pts_ext, lin2.predict(poly.fit_transform(Ic_pts_ext)), color = 'red') 
plt.plot(Ic, Eon, color = 'green', linestyle='dashed') 
plt.title('Polynomial Regression (Polynomial Testing w/ a fit by hand)') 
plt.xlabel('$I_c$ [A]') 
plt.ylabel('$E_{on}$ [J]')
plt.show() 

#%%
#%%
#%%
#%%

poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(Ic_pts)

poly.fit(X_poly, E_off_pts)

lin2 = LinearRegression()
lin2.fit(X_poly, E_off_pts) 

plt.scatter(Ic_pts, E_off_pts, color = 'blue') 
plt.plot(Ic_pts_ext, lin2.predict(poly.fit_transform(Ic_pts_ext)), color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('$I_c$ [A]') 
plt.ylabel('$E_{off}$ [J]')
plt.show() 

print(lin2.coef_)

#%%

Ic = Ic_pts_ext
Eoff = .05 + 1.86104115e-03*Ic -1.95635972e-07*Ic**2

plt.plot(Ic_pts_ext, lin2.predict(poly.fit_transform(Ic_pts_ext)), color = 'red') 
plt.plot(Ic, Eoff, color = 'green', linestyle='dashed') 
plt.title('Polynomial Regression') 
plt.xlabel('$I_c$ [A]') 
plt.ylabel('$E_{on}$ [J]')
plt.show()

#%%

plt.plot(Ic, Eon, color = 'black', label='$E_{on}$') 
plt.plot(Ic, Eoff, color = 'green', label='$E_{off}$') 
plt.title('Polynomial Regression (DataSheet compare)')
plt.xlabel('$I_c$ [A]') 
plt.ylabel('$E_{on}$ $E_{off}$ [J]')
plt.legend()
plt.grid()
plt.show()

#%% Erec ready

Ic_pts_ext = np.linspace(0, 3000, 3000)
Ic_pts_ext = Ic_pts_ext.reshape(Ic_pts_ext.size, 1)

Ic = Ic_pts_ext

Erec = -3.45e-7 * Ic**2 + 1.45e-3 * Ic + 285e-3

plt.plot(Ic, Erec, color = 'green', linestyle='dashed') 
plt.title('Polynomial Regression') 
plt.xlabel('$I_c$ [A]') 
plt.ylabel('$E_{rec}$ [J]')
plt.grid()
plt.show()

#%%

def poly_energy(current, params=[1,1,1,1]):
    
    ret = np.zeros(current.size)
    n_p = len(params) - 1
    
    for c, p in enumerate(params):
        # print('p', p)
        # print('n_p - c', n_p - c)
        ret += p * current ** (n_p - c)
        
    return ret

eon_params  = [5.48091255e-07, 5.63340539e-04, .4]
eoff_params = [-1.95635972e-07, 1.86104115e-03, .05]
erec_params = [-3.45e-7, 1.45e-3, 285e-3]

i = np.linspace(0, 3000, 200)

eon = poly_energy(i, eon_params)
eoff = poly_energy(i, eoff_params)
erec = poly_energy(i, erec_params)

print('eon_params:', eon_params)
print('eoff_params:', eoff_params)
print('erec_params:', erec_params)

plt.plot(i, eon, label='Poly On')
plt.plot(i, eoff, label='Poly Off')
plt.plot(i, erec, label='Poly Rec')
plt.title('Polynomial Regression') 
plt.xlabel('$I_c$ [A]') 
plt.ylabel('$E_{on}$, $E_{off}$, $E_{rec}$ [J]')
plt.grid()
plt.legend()
plt.show()

channels = [  
    'Three Phase Inverter1.IGBT Leg1.il',
    'Three Phase Inverter1.IGBT Leg1.v1',
    'Three Phase Inverter1.IGBT Leg1.v2',
    
    'Three Phase Inverter1.IGBT Leg2.il',
    'Three Phase Inverter1.IGBT Leg2.v1',
    'Three Phase Inverter1.IGBT Leg2.v2',
    
    'Three Phase Inverter1.IGBT Leg3.il',
    'Three Phase Inverter1.IGBT Leg3.v1',
    'Three Phase Inverter1.IGBT Leg3.v2',
    
    'Three Phase Inverter1.IGBT Leg1.PWM_Modulator_TOP_1',
    'Three Phase Inverter1.IGBT Leg2.2',
    'Three Phase Inverter1.IGBT Leg3.PWM_Modulator_TOP_1',
    
    'Three Phase Inverter1.IGBT Leg1_stf',
    'Three Phase Inverter1.IGBT Leg2_stf',
    'Three Phase Inverter1.IGBT Leg3_stf']

hil.start_simulation()

cap.wait(time_to_ss)

_, _, rate = cap.start_capture(duration=total_capture_time,
                               rate=1//cap_step_exp,
                               signals=channels)

df1 = cap.get_capture_results(wait_capture=True)

hil.stop_simulation()

#%%

df1C = df1.copy()
df1C.index /= np.timedelta64(1, 's')

#%%
import matplotlib.pyplot as plt

ilA = df1C['Three Phase Inverter1.IGBT Leg1.il']
ilB = df1C['Three Phase Inverter1.IGBT Leg2.il']
ilC = df1C['Three Phase Inverter1.IGBT Leg3.il']

plt.plot(df1C.index, ilA, df1C.index, ilB, df1C.index, ilC)
plt.xlim(0, 0.1)
plt.show()

#%%
v1A = df1C['Three Phase Inverter1.IGBT Leg1.v1']
v1B = df1C['Three Phase Inverter1.IGBT Leg2.v1']
v1C = df1C['Three Phase Inverter1.IGBT Leg3.v1']

plt.plot(df1C.index, v1A, df1C.index, v1B + 850, df1C.index, v1C + 1700)
plt.xlim(0, 0.0125/16)
plt.show()

v1A = df1C['Three Phase Inverter1.IGBT Leg1.v2']
v1B = df1C['Three Phase Inverter1.IGBT Leg2.v2']
v1C = df1C['Three Phase Inverter1.IGBT Leg3.v2']

plt.plot(df1C.index, v1A, df1C.index, v1B + 850, df1C.index, v1C + 1700)
plt.xlim(0, 0.0125/16)
plt.show()

v1A = df1C['Three Phase Inverter1.IGBT Leg1.v1']
v1B = df1C['Three Phase Inverter1.IGBT Leg1.v2']

plt.plot(df1C.index, v1A, df1C.index, v1B + 850)
plt.xlim(0, 0.0125/16)
plt.show()

v1A = df1C['Three Phase Inverter1.IGBT Leg2.v1']
v1B = df1C['Three Phase Inverter1.IGBT Leg2.v2']

plt.plot(df1C.index, v1A, df1C.index, v1B + 850)
plt.xlim(0, 0.0125/16)
plt.show()

v1A = df1C['Three Phase Inverter1.IGBT Leg3.v1']
v1B = df1C['Three Phase Inverter1.IGBT Leg3.v2']

plt.plot(df1C.index, v1A, df1C.index, v1B + 850)
plt.xlim(0, 0.0125/16)
plt.show()

v1A = df1C['Three Phase Inverter1.IGBT Leg1.v2']
v1B = df1C['Three Phase Inverter1.IGBT Leg2.v2']
v1C = df1C['Three Phase Inverter1.IGBT Leg3.v2']

plt.plot(df1C.index, v1A, df1C.index, v1B + 850, df1C.index, v1C + 1700)
plt.xlim(0, 0.0125/16)
plt.show()

v1A = df1C['Three Phase Inverter1.IGBT Leg1.PWM_Modulator_TOP_1']
v1B = df1C['Three Phase Inverter1.IGBT Leg2.PWM_Modulator_TOP_1']
v1C = df1C['Three Phase Inverter1.IGBT Leg3.PWM_Modulator_TOP_1']

plt.plot(df1C.index, v1A, df1C.index, v1B + 1.5, df1C.index, v1C + 3)
plt.xlim(0, 0.0125/16)
plt.show()

v1A = df1C['Three Phase Inverter1.IGBT Leg1_stf']
v1B = df1C['Three Phase Inverter1.IGBT Leg2_stf']
v1C = df1C['Three Phase Inverter1.IGBT Leg3_stf']

plt.plot(df1C.index, v1A, df1C.index, v1B + 1.5, df1C.index, v1C + 3)
#plt.xlim(0, 0.0125/16)
plt.show()

#%%
v1A = df1C['Three Phase Inverter1.IGBT Leg1.PWM_Modulator_TOP_1']
v1B = df1C['Three Phase Inverter1.IGBT Leg1.v1']
v1C = df1C['Three Phase Inverter1.IGBT Leg1.v2']

plt.subplot(311)
plt.plot(df1C.index, v1A)
plt.xlim(0, 0.0125/16)
plt.subplot(312)
plt.plot(df1C.index, v1B)
plt.xlim(0, 0.0125/16)
plt.subplot(313)
plt.plot(df1C.index, v1C)
plt.xlim(0, 0.0125/16)

plt.show()

v1A = df1C['Three Phase Inverter1.IGBT Leg2.PWM_Modulator_TOP_1']
v1B = df1C['Three Phase Inverter1.IGBT Leg2.v1']
v1C = df1C['Three Phase Inverter1.IGBT Leg2.v2']

plt.subplot(311)
plt.plot(df1C.index, v1A)
plt.xlim(0, 0.0125/16)
plt.subplot(312)
plt.plot(df1C.index, v1B)
plt.xlim(0, 0.0125/16)
plt.subplot(313)
plt.plot(df1C.index, v1C)
plt.xlim(0, 0.0125/16)

plt.show()

v1A = df1C['Three Phase Inverter1.IGBT Leg3.PWM_Modulator_TOP_1']
v1B = df1C['Three Phase Inverter1.IGBT Leg3.v1']
v1C = df1C['Three Phase Inverter1.IGBT Leg3.v2']

plt.subplot(311)
plt.plot(df1C.index, v1A)
plt.xlim(0, 0.0125/16)
plt.subplot(312)
plt.plot(df1C.index, v1B)
plt.xlim(0, 0.0125/16)
plt.subplot(313)
plt.plot(df1C.index, v1C)
plt.xlim(0, 0.0125/16)

plt.show()