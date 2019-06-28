# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:50:13 2019

@author: tiarl
"""

#%%
# Swithching Power e Conduction Power

import numpy as np
import matplotlib.pyplot as plt


def poly_energy(current, params=[1,1,1,1]):
    
    ret = np.zeros(current.size)
    n_p = len(params) - 1
    
    for c, p in enumerate(params):
        # print('p', p)
        # print('n_p - c', n_p - c)
        ret += p * current ** (n_p - c)
        
    return ret


def detectCommutation(y):
    y1 = y[:-1]
    y2 = y[1:]
    
    diff = y2 - y1
    
    return y1, y2, diff


def tableSwitchingLossesJoule(switch_s1, current, duration, step, eon_params=[1, 1, 1, 1],
                         eoff_params=[1, 1, 1, 1], erec_params=[1, 1, 1, 1]):
    '''
        Table with Switching Losses for TDD 2-Levels.
    '''
    _, _, commut = detectCommutation(switch_s1)
    
    commut_pos = commut == 1
    commut_neg = commut == -1
    
    current0 = current[1:]
    
    current_pos = current0 >= 0
    current_neg = current0 < 0
    
    comut_neg_cur_pos = commut_neg * current_pos
    comut_neg_cur_neg = commut_neg * current_neg
    comut_pos_cur_pos = commut_pos * current_pos
    comut_pos_cur_neg = commut_pos * current_neg
    
    current_comut_neg_cur_pos = np.abs(current0[comut_neg_cur_pos])
    current_comut_neg_cur_neg = np.abs(current0[comut_neg_cur_neg])
    current_comut_pos_cur_pos = np.abs(current0[comut_pos_cur_pos])
    current_comut_pos_cur_neg = np.abs(current0[comut_pos_cur_neg])

    # For 'S1'
    s1_eoff = poly_energy(current_comut_neg_cur_pos, params=eoff_params)
    s1_eon = poly_energy(current_comut_pos_cur_pos, params=eon_params)
    
    # For 'S2'
    s2_eoff = poly_energy(current_comut_pos_cur_neg, params=eoff_params)
    s2_eon = poly_energy(current_comut_neg_cur_neg, params=eon_params)
    
    # For 'D1'
    d1_rec = poly_energy(current_comut_neg_cur_neg, params=erec_params)
    
    # For 'D2'
    d2_rec = poly_energy(current_comut_pos_cur_pos, params=erec_params)
    
    P_sw = {
        'S1': 1/duration * (np.sum(s1_eoff) + np.sum(s1_eon)) * step,
        'S2': 1/duration * (np.sum(s2_eoff) + np.sum(s2_eon)) * step,
        'D1': 1/duration * (np.sum(d1_rec)) * step,
        'D2': 1/duration * (np.sum(d2_rec)) * step}

    return P_sw, ((commut_pos, commut_neg, current_pos, current_neg,
            (comut_neg_cur_pos, comut_neg_cur_neg), 
            (comut_pos_cur_pos, comut_pos_cur_neg), 
            (current_comut_neg_cur_pos, current_comut_neg_cur_neg),
            (current_comut_pos_cur_pos, current_comut_pos_cur_neg)))

#%%
# TESTING
y = np.random.randint(0, 2, size=10)
y1, y2, diff = detectCommutation(y)

plt.subplot(411)
plt.step(range(10), y, where='post', label='y')
plt.xlim(0,9)
plt.legend()
plt.subplot(412)
plt.step(range(1, 10), y1, where='post', label='y[1:]')
plt.xlim(0,9)
plt.legend()
plt.subplot(413)
plt.step(range(1,10), y2, where='post', label='y[:-1]')
plt.xlim(0,9)
plt.legend()
plt.subplot(414)
plt.plot(range(1, 10), diff, 'ko ', label='commut detect')
plt.xlim(0,9)
plt.legend()
plt.show()

#%%
size = 30

np.random.seed(5)
t = np.linspace(0, 1/60, size)
sim_switch  = np.random.randint(0, 2, size=size)
sim_current = np.random.randint(-311, 312, size=size)

#%%
plt.subplot(211)
plt.step(range(sim_switch.size), sim_switch, label='sim_switch', where='post')
plt.plot(range(sim_switch.size), sim_switch, 'ko ', label='sim_switch')
plt.grid(True)
plt.legend()
plt.subplot(212)
plt.plot(sim_current, label='sim_current')
plt.grid(True)
plt.legend()
plt.show()
#%%

#Eon = .4 + 5.63340539e-04*Ic + 5.48091255e-07*Ic**2
#Eoff = .05 + 1.86104115e-03*Ic -1.95635972e-07*Ic**2
#Erec = -3.45e-7 * Ic**2 + 1.45e-3 * Ic + 285e-3


eon_params = [1,1,1,0]
eoff_params = [1,1,1,10000000]
erec_params = [1,1,1,20000000]

eon_params  = [5.48091255e-07, 5.63340539e-04, .4]
eoff_params = [-1.95635972e-07, 1.86104115e-03, .05]
erec_params = [-3.45e-7, 1.45e-3, 285e-3]

print('eon_params:', eon_params)
print('eoff_params:', eoff_params)
print('erec_params:', erec_params)

P_sw, tmp = tableSwitchingLossesJoule(sim_switch, sim_current, t[-1], t[1] - t[0], eon_params=eon_params, eoff_params=eoff_params, erec_params=erec_params)
a1, a2, b1, b2, c1, c2, d1, d2 = tmp

#%%

plt.figure(figsize=(5, 10))
plt.subplot(611)
plt.plot(range(size), sim_switch, 'bo')
plt.step(range(size), sim_switch, where='post', label='Commut')
plt.xlim(0, size)
plt.legend()

plt.subplot(612)
plt.plot(range(size), sim_current, 'bo-', label='Current [A]')
plt.xlim(0, size)
plt.grid(True)
plt.legend()

plt.subplot(613)
plt.plot(range(1, size), a1, 'bo', label='Pos Commut')
plt.xlim(0, size)
plt.legend()

plt.subplot(614)
plt.plot(range(1, size), a2, 'ro', label='Neg Commut')
plt.xlim(0, size)
plt.legend()

plt.subplot(615)
plt.plot(range(1, size), b1, 'bo', label='Pos Current')
plt.xlim(0, size)
plt.legend()

plt.subplot(616)
plt.plot(range(1, size), b2, 'ro', label='Neg Current')
plt.xlim(0, size)
plt.legend()

plt.tight_layout()
plt.show()

#%%

plt.subplot(411)
plt.plot(range(1, size), c1[0], 'bo', label='comut_neg_cur_pos')
plt.xlim(0, size)
plt.legend(framealpha=.3)

plt.subplot(412)
plt.plot(range(1, size), c1[1], 'ro', label='comut_neg_cur_neg')
plt.xlim(0, size)
plt.legend(framealpha=.3)

plt.subplot(413)
plt.plot(range(1, size), c2[0], 'go', label='comut_pos_cur_pos')
plt.xlim(0, size)
plt.legend(framealpha=.3)

plt.subplot(414)
plt.plot(range(1, size), c2[1], 'ko', label='comut_pos_cur_neg')
plt.xlim(0, size)
plt.legend(framealpha=.3)
plt.tight_layout()
plt.show()

#%%
plt.figure(figsize=(5, 5))
plt.subplot(411)
plt.plot(d1[0], 'bo', label='current_comut_neg_cur_pos')
#plt.xlim(0, size)
plt.legend(framealpha=.3)

plt.subplot(412)
plt.plot(d1[1], 'ro', label='current_comut_neg_cur_neg')
#plt.xlim(0, size)
plt.legend(framealpha=.3)

plt.subplot(413)
plt.plot(d2[0], 'go', label='current_comut_pos_cur_pos')
#plt.xlim(0, size)
plt.legend(framealpha=.3)

plt.subplot(414)
plt.plot(d2[1], 'ko', label='current_comut_pos_cur_neg')
#plt.xlim(0, size)
plt.legend(framealpha=.3)
plt.tight_layout()
plt.show()

#%%

plt.plot([0], P_sw['S1'], 'o', label='Power Switching (S1)')
plt.plot([1], P_sw['S2'], 'o', label='Power Switching (S2)')
plt.plot([2], P_sw['D1'], 'o', label='Power Switching (D1)')
plt.plot([3], P_sw['D2'], 'o', label='Power Switching (D2)')
plt.grid()
plt.legend()
plt.show()

#%%
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

#%%

def rmsValue(current):
    '''
    Calculate the RMS value.
    '''
    
    import numpy as np
    
    return np.sqrt(np.sum(np.power(current, 2)) / current.size)


def tableConductionLossesJoule(switch_s1, current, Vout):#, 
    '''
        Table with Switching Losses for TDD 2-Levels.
    '''
    
    switch_s2 = np.zeros(switch_s1.size)
    switch_s2 = switch_s1 == 0
    
    pos_current = current >= 0
    neg_current = current < 0

    s1_times_current = switch_s1 * current
    s2_times_current = switch_s2 * current

    current_s1 = np.zeros(current.size)
    current_s1[pos_current] = s1_times_current[pos_current]

    current_d1 = np.zeros(current.size)
    current_d1[neg_current] = -1 * s1_times_current[neg_current]

    current_s2 = np.zeros(current.size)
    current_s2[neg_current] = -1 * s2_times_current[neg_current]

    current_d2 = np.zeros(current.size)
    current_d2[pos_current] = s2_times_current[pos_current]
    
    s1_cd = Vout * ( current_s1 + np.power(rmsValue(current_s1), 2) )
    s2_cd = Vout * ( current_s2 + np.power(rmsValue(current_s2), 2) )
    d1_cd = Vout * ( current_d1 + np.power(rmsValue(current_d1), 2) )
    d2_cd = Vout * ( current_d2 + np.power(rmsValue(current_d2), 2) )
    
    P_cd = {'S1': s1_cd,
            'S2': s2_cd,
            'D1': d1_cd,
            'D2': d2_cd}
    
    return P_cd, ((switch_s1, switch_s2), 
                  (current_s1, current_d1, current_s2, current_d2),
                  (rmsValue(current_s1), rmsValue(current_d1), 
                   rmsValue(current_s2), rmsValue(current_d2)))

#%%
# TESTING CONDUCTION LOSSES

size = 30

t = np.linspace(0, 1/50, size)

np.random.seed(5)
sim_switch_s1  = np.random.randint(0, 2, size=size)
sim_current = 311*np.sin(2*np.pi*50*t)

Vout = 220

P_cd, tmp = tableConductionLossesJoule(sim_switch_s1, sim_current, Vout)

((switch_s1, switch_s2), 
 (current_s1, current_d1, current_s2, current_d2), 
 (current_s1_rms, current_d1_rms, 
  current_s2_rms, current_d2_rms)) = tmp

#%%

plt.subplot(211)
plt.step(t, switch_s1, label='Switch 1', where='post')
plt.plot(t, switch_s1, 'ko ', label='Switch 1')
plt.legend()
plt.grid()
plt.subplot(212)
plt.step(t, switch_s2, label='Switch 2', where='post')
plt.plot(t, switch_s2, 'ko ', label='Switch 2')
plt.legend()
plt.grid()

plt.show()

#%%

fig, ax = plt.subplots(5,2)

# plt.subplot(521)
ax[0][0].plot(t, sim_current, 'bo-', label='Current')
ax[0][0].legend()
ax[0][0].grid()
# plt.subplot(522)
ax[0][1].plot(t, sim_current, 'bo-', label='Current')
ax[0][1].legend()
ax[0][1].grid()

# plt.subplot(523)
ax[1][0].step(t, switch_s1, label='Current', where='post')
ax[1][0].plot(t, switch_s1, 'ko ', label='Current')
ax[1][0].legend()
ax[1][0].grid()
# plt.subplot(524)
ax[1][1].step(t, switch_s1, label='Current', where='post')
ax[1][1].plot(t, switch_s1, 'ko ', label='Current')
ax[1][1].legend()
ax[1][1].grid()

# plt.subplot(525)
ax[2][0].plot(t, current_s1, 'go ', label='Current S1')
ax[2][0].legend()
ax[2][0].grid()
# plt.subplot(526)
ax[2][1].plot(t, current_s2, 'go ', label='Current S2')
ax[2][1].legend()
ax[2][1].grid()

# plt.subplot(527)
ax[3][0].plot(t, current_d2, 'ro ', label='Current D2')
ax[3][0].legend(framealpha=.2)
ax[3][0].grid()
# plt.subplot(528)
ax[3][1].plot(t, current_d1, 'ro ', label='Current D1')
ax[3][1].legend(framealpha=.2)
ax[3][1].grid()

# plt.subplot(529)
ax[4][0].plot(t, current_s1, 'go ', label='Current S1')
ax[4][0].plot(t, current_d2, 'ro ', label='Current D2')
ax[4][0].legend(framealpha=.2)
ax[4][0].grid()
# plt.subplot(5210)
ax[4][1].plot(t, current_s2, 'go ', label='Current S2')
ax[4][1].plot(t, current_d1, 'ro ', label='Current D1')
ax[4][1].legend(framealpha=.2)
ax[4][1].grid()

# plt.tight_layout()
plt.show()

#%%

plt.plot([0], current_s1_rms, 'o', label='$I_{S1, RMS}$')
plt.plot([1], current_d1_rms, 'o', label='$I_{D1, RMS}$')
plt.plot([2], current_s2_rms, 'o', label='$I_{S2, RMS}$')
plt.plot([3], current_d2_rms, 'o', label='$I_{D2, RMS}$')
plt.legend()
plt.grid()
plt.show()

#%%

plt.plot(current_s1)
plt.plot(current_d2, 'g--')
plt.show()