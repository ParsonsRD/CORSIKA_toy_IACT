import numpy as np
import matplotlib.pyplot as plt
#
EPOS_pass_100GeV = 0.37589642433259224
QGS_pass_100GeV = 0.37659150721957463
SIBYLL_pass_100GeV = 0.3723415349745886

EPOS_pass_1TeV = 0.11207189756314036
QGS_pass_1TeV = 0.10531294548798195
SIBYLL_pass_1TeV = 0.11078150837122779

EPOS_pass_10TeV = 0.06105078323552516
QGS_pass_10TeV = 0.050561018250437066
SIBYLL_pass_10TeV = 0.06488577881446925

energy = [0.1,1,10] #TeV
EPOS = [EPOS_pass_100GeV, EPOS_pass_1TeV, EPOS_pass_10TeV]
QGS = [QGS_pass_100GeV, QGS_pass_1TeV, QGS_pass_10TeV]
SIBYLL = [SIBYLL_pass_100GeV, SIBYLL_pass_1TeV, SIBYLL_pass_10TeV]

plt.plot(energy,EPOS, label='EPOS', color='red')
plt.plot(energy,QGS, label='QGS', color = 'green')
plt.plot(energy,SIBYLL, label='SIBYLL', color = 'blue')
plt.grid()
plt.xscale('log')
plt.title('Model Comparison')
plt.xlabel('Energy in TeV')
plt.ylabel('Passing  Fraction')
plt.legend()
plt.show()
