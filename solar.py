import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

#Ap: sol+terra+marte
#1. cond iniciais -> centro massa -> var. adims. ->
#N corpos -> 3N x+3N v em cada t

#r1(t0)=(0,0,0),v1(t0)=(0,0,0)
#r2(t0)=(d2,0,0), v2(t0)=(0,v2,0)
#r3(t0)=(d3,0,0) v3(t0)=(0,v3,0)

M_sol=1.989e30 #kg
M_t=5.972e24 #kg
M_m=6.39e23 #kg
d2=1 * 1.49597e11 #m, x_terra
d3=1.524 * 1.49597e11 #m, marte
#v2=np.sqrt(G*(ms[0]+ms[1)/ds[1]) #cm*s⁻¹, v_rel terra-sol
G=6.67408e-11 #m³kg⁻¹s⁻²
T_sol,T_terra,T_marte=0,365*24*3600,686*24*3600 #s

def ini_conds1(N,ds,ms,Ts): #1->sol,2->terra,3->marte
    "rs=(0,d2,d3), onde r1=(0,0,0), r2=(d2,0,0), r3=(d3,0,0)"
    "vs=(0,v2,v3), onde v1=(0,0,0), v2=(0,v2,0), v3=(0,v3,0)"
    "ms=(m1,m2,m3)"

    v_i, vs=np.zeros(N, dtype=float), np.zeros((N,N), dtype=float)
    rs=vs*0.
    rs[:,0]=ds
    for i in range(1,N): #v[0]=0
        v_i[i]=np.pi*2*ds[i]/Ts[i] #ms⁻¹
        vs[i,1]=v_i[i]
    R=max(ds)
    M=np.sum(ms)
    return M,R,rs,vs,ms

M_x,R_x,rs_x,vs_x,m_x=ini_conds1(3,np.array([0,d2,d3]),np.array([M_sol,M_t,M_m]),np.array([T_sol,T_terra,T_marte]))

def cm_va(M,R,rs,vs,m):
    r_cm=(np.dot(m[None, :],rs))/M
    v_cm=(np.dot(m[None, :],vs))/M
    v_i, r_i=vs-v_cm, rs-r_cm
    x_adim, m_adim=r_i/R, m/M
    y_adim=v_i/np.sqrt(G*M/2/R)
    print(y_adim)
    return m_adim,x_adim,y_adim

m_d,x_d,y_d=cm_va(M_x,R_x,rs_x,vs_x,m_x)

def bi(x,m,epsilon=1e-10):
    N=len(x)
    b = np.zeros((N,3))
    for i in range(N): #fixar particula
        for j in range(N): #outra particula
            if i != j:
                d=np.linalg.norm((x[i]-x[j]))**3
                b[i,:]+=(m[j]*(x[i]-x[j])/d)+epsilon #força entre particulas
    return b

b_i=bi(x_d,m_d)
r_0=np.concatenate((x_d.flatten(),y_d.flatten()),axis=0)

def ncorpos(u,r):
    "Define a equação diferencial a resolver"
    N=len(m_d)
    xs,ys=np.reshape(r[:3*N],(N,3)),np.reshape(r[3*N:],(N,3))
    dx=2*ys
    dy=-4*bi(xs,m_d)
    rf=np.concatenate((dx.flatten(),dy.flatten()),axis=0)
    return rf


def sol_eq(r0,T,step):
    "Resolve a equação e retorna 3 arrays 1d: evolução de todas as componentes das posições, evolução das componentes das velocidades, tempos"
    N=len(m_d)
    xs=np.zeros((N,3),dtype=float)
    ys=xs*0.
    sol=spi.solve_ivp(ncorpos,[0,T],r0,max_step=step)
    ts=sol.t
    r=sol.y
    xs,ys=r[:3*N,:],r[3*N:,:]
    return xs,ys/np.max(ys),ts

def energy_v(m,x,y):
    N = len(m)
    v=y.reshape(N,3)
    KE=np.sum(m*np.sum(v**2,axis=1))
    r=x.reshape(N, 3)
    PE=0
    for i in range(N):
        for j in range(N):
            if i != j:
                d=np.linalg.norm((r[i]-r[j]))
                PE-=.5*((m[i]*m[j])/d)
    return PE,KE

def get_energy(m,x,y,ts):
    KE=np.zeros(len(ts))
    PE=np.copy(KE)
    for t in range(len(ts)):
        PE[t],KE[t]=energy_v(m,x[:,t],y[:,t])
    return PE,KE



xs,ys,ts=sol_eq(r_0,200,0.05)
xs_1,ys_1,ts_1=sol_eq(r_0,2,0.04)

xs_s,ys_s,ts_s=sol_eq(r_0,10,0.08)
xs_s2,ys_s2,ts_s2=sol_eq(r_0,10,0.04)



PE1,KE1=get_energy(m_d,xs_s,ys_s,ts_s)
PE2,KE2=get_energy(m_d,xs_s2,ys_s2,ts_s2)




plt.plot(xs[0,:],xs[1,:], '.', label="Sun")
plt.plot(xs[3,:],xs[4,:], label="Earth")
plt.plot(xs[6,:],xs[7,:], label="Mars")
plt.legend(fontsize=13)
plt.savefig("orbits.pdf",dpi=1000,transparent=True,bbox_inches='tight')
plt.show()


plt.plot(ts_1,ys_1[3,:], label="vx Earth")
plt.plot(ts_1,ys_1[4,:], label="vy Earth")
plt.plot(ts_1,ys_1[6,:], label="vx Mars")
plt.plot(ts_1,ys_1[7,:], label="vy Mars")
plt.xlabel("time",fontsize=13)
plt.ylabel("velocity, normalized", fontsize=13 )
plt.legend(fontsize=13)
plt.savefig("velocity.pdf",dpi=1000,transparent=True,bbox_inches='tight')
plt.legend()
plt.show()


plt.plot(ts_s,KE1+PE1,label="step=0.08")
plt.xlabel("time", fontsize=13)
plt.ylabel("total energy, arbitrary u4its",fontsize=13)
plt.plot(ts_s2,KE2+PE2,label="step=0.04")
plt.legend(fontsize=13)
plt.savefig("energy.pdf",dpi=1000,transparent=True,bbox_inches='tight')
plt.show()

