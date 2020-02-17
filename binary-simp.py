import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

## Sistema Alpha Centauri: Alpha Centauri A, B, Proxima + corpo
G=6.67408e-11 #m³kg⁻¹s⁻1
M_s=1.989e30 #kg
AU= 1.49597e11 #m
pc=3.08567758e16 #m

M_A, M_B, M_P, M_c =1.1055*M_s, 0.9373*M_s,0.1221*M_s,0 #kg

#distâncias ao baricentro
d_A,d_B,d_P=10.9*AU,12.8*AU,13*10e3*AU #m, semieixo maior

#condições iniciais fixas - coordenadas heliocentricas
pos_AB=np.array([.95845*pc,-.93402*pc,-.01601*pc])
pos_P=np.array([.90223*pc,-.93599*pc,-.04386*pc])
v_AB, v_P=np.array([-29291,1710,13589]), np.array([-29390,1883,13777]) #ms⁻¹

vs_sis2=np.reshape(np.concatenate((v_AB,v_P)),(2,3))
pos_sis2=np.reshape(np.concatenate((pos_AB,pos_P)),(2,3))

ms_d1=np.array([M_A+M_B,M_P])
R_d=np.max(np.array([np.max(np.absolute(pos_AB)),np.max(np.absolute(pos_P))]))
M_d=np.sum(ms_d1)


##################solver#########################

def cm_va(M,R,rs,vs,m):
    r_cm=(np.dot(m[None, :],rs))/M
    v_cm=(np.dot(m[None, :],vs))/M
    v_i, r_i=vs-v_cm, rs-r_cm
    x_adim, m_adim=r_i/R, m/M
    y_adim=v_i/np.sqrt(G*M/2/R)
    return m_adim,-x_adim/np.absolute(x_adim).max(),y_adim/np.absolute(y_adim).max()

def bi(x,m):
    N=len(x)
    b = np.zeros((N,3))
    for i in range(N): #fixar particula
        for j in range(N): #outra particula
            if i != j:
                d=np.linalg.norm((x[i]-x[j]))**3
                b[i,:]+=(m[j]*(x[i]-x[j])/(d + 1e-16)) #força entre particulas
    return b

def ncorpos(u,r,m):
    "Define a equação diferencial a resolver"
    N=len(m)
    xs,ys=np.reshape(r[:3*N],(N,3)),np.reshape(r[3*N:],(N,3))
    dx=2*ys
    dy=-4*bi(xs,m)
    rf=np.concatenate((dx.flatten(),dy.flatten()),axis=0)
    return rf

def sol_eq(r0,m,T,step=0.0001):
    "Resolve a equação e retorna 3 arrays 1d: evolução de todas as componentes das posições, evolução das componentes das velocidades, tempos"
    N=len(m)
    xs=np.zeros((N,3),dtype=float)
    ys=xs*0.
    sol=spi.solve_ivp(lambda t,r: ncorpos(t,r,m),[0,T],r0,max_step=step)
    ts=sol.t
    r=sol.y
    xs,ys=r[:3*N,:],r[3*N:,:]
    return xs/xs.max(),ys/ys.max(),ts

def energy_v(m,x,y,p=0):
    N = len(m)
    v=y.reshape(N,3)
    r=x.reshape(N,3)
    if p!=0:
        v=v[:1,:]
        r=r[:1,:]
        m=m[:1]
        N=len(m)
    KE=np.sum(m*np.sum(v**2,axis=1))
    PE=0
    for i in range(N):
        for j in range(N):
            if i != j:
                d=np.linalg.norm((r[i]-r[j]))
                PE-=.5*((m[i]*m[j])/d)
    return PE,KE

def get_energy(m,x,y,ts,p=0):
    KE=np.zeros(len(ts))
    PE=np.copy(KE)
    for t in range(len(ts)):
        PE[t],KE[t]=energy_v(m,x[:,t],y[:,t],p)
    return PE,KE


#
###fazer  plot para massas diferentes com condições iniciais
#[4,0,0]
#[4,-2,0]
#[4,-4,0]

##tests###
#condições iniciais sem corpo extra nas coordenadas do CM do sistema alfa centauri
m_d1,x_d1,y_d1=cm_va(M_d,R_d,pos_sis2,vs_sis2,ms_d1)

##conds iniciais para o corpo extra - relativo ao cm d 
x_corpo=np.array([4, 0 ,0])
y_corpo=np.array([-1,1,0])

m_corpo=26
#m_list=[18,19,20]

x_new,y_new=np.append(x_d1,x_corpo),np.append(y_d1,y_corpo)

r_2=np.concatenate((x_d1.flatten(),y_d1.flatten()),axis=0)
r_3=np.concatenate((x_new.flatten(),y_new.flatten()),axis=0)

#x2,y2,t2=sol_eq(r_2,m_d1,500,0.005)

m_new=np.append(m_d1,m_corpo)
#for m in m_list:
#m_new=np.append(m_d1,m)
xs3,ys3,ts3=sol_eq(r_3,m_new,50,0.01)

#PE3,KE3=get_energy(m_new,xs3,ys3,ts3,1)
#PE1,KE1=get_energy(m_d1,x2,y2,t2,1)

#plt.plot(x2[0,:],x2[1,:], label=r"$\alpha$ A+B")
#plt.plot(x2[3,:],x2[4,:], label=r"$\alpha$ P")
#plt.legend(fontsize=13)
#plt.savefig("xy-bin.pdf",dpi=1000,transparent=True,bbox_inches='tight')
#plt.title("arbitrary plane of motion")
#plt.show()

T=len(ts3)
t=T//10
for i in range(10):
    plt.plot(xs3[0,:][:i*t],xs3[1,:][:i*t], label=r"$\alpha$ A+B")
    plt.plot(xs3[3,:][:i*t],xs3[4,:][:i*t], label=r"$\alpha$ P")
    plt.plot(xs3[6,:][:i*t],xs3[7,:][:i*t],label=r"$M/M_T $ = {}".format(m_corpo))
    plt.legend(fontsize=13)
    plt.savefig("perturb001.pdf",dpi=1000,transparent=True,bbox_inches='tight')
    plt.xlabel("arbitrary plane: axis 1", fontsize=15)
    plt.ylabel("arbitrary plane: axis 2", fontsize=15)
    plt.show()



    ####energy
   # plt.plot(ts3,KE3+PE3,label=r"added  body, $M/M_T = ${}".format(m_corpo))
   # plt.xlabel("time", fontsize=15)
   # plt.ylabel("total energy, arbitrary units",fontsize=15)
   # plt.savefig("energy001.pdf",dpi=1000,transparent=True,bbox_inches='tight')
  #  plt.plot(t2,KE1+PE1,label="original system")
 #   plt.legend(fontsize=13)
#    plt.show()

