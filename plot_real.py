import numpy as np
import matplotlib.pyplot as plt

w_l2 = [-1.761,0.5099,0.04078,0.09048,0.02809,0.8424,-0.9509,-0.09409,\
        -0.3489,-0.08853,0.2360,0.6620,0.01256,0.4007,-0.09985]

w_vi = [-1.738,0.5101,0.04079,0.09054,0.02804,0.8429,-0.9518,-0.09412,\
        -0.3491,-0.08866,0.2361,0.9991,0.01059,0.4010,-0.1001]

w_sv = [-2.171,0.5732,0.04634,0.1049,0.01732,0.9572,-1.190,-0.1034,\
        -0.4205,-0.1478,0.2975,1.042,-1.7451,0.4986,-0.3569]

x = range(15)
fig, ax=plt.subplots()
ax.plot(x,w_l2,'k',label='L2 Regularization')
ax.plot(x,w_vi,'b',label='Regular Variational')
ax.plot(x,w_sv,'r',label='Stochastic Variational')

plt.xlabel("Weights for each feature")
plt.ylabel("Values of weights")
plt.title("Comparision of Weights Values between algorithms")
#plt.legend(('l2','vi','sv'),loc='upper right')
legend = ax.legend(loc='upper left')
plt.show()

