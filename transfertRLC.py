import matplotlib.pyplot as plt
import numpy as np
import mysounds as ms

def db(x):
	return 10*np.log(x)

def RLC(R,L,C,f):
	RC2 = R*R*C*C
	w0 = 1.0/np.sqrt(L*C)
	w02 = w0*w0
	Q = w0*L/R
	Q2 = Q*Q
	w = 2*np.pi*f
	u2 = w*w/w02
	v = w*1j/w0
	G = 1.0/np.sqrt((1-u2)**2 + (u2/Q2))
	H = 1.0/(1 + v/Q + v**2)
	#G = G/(1 + R*C*w)
	f0 = w0/(2*np.pi)
    #H = 1.0/(1 + R*C*w)
	return f0,Q,G,H


def sklowpass(R1,R2,C1,C2,f):
	w = 2*np.pi*f
	w0 = 1.0/np.sqrt(R1*R2*C1*C2)
	w02 = w0*w0
	u2 = w*w/w02
	
	b = (1.0/C1)*((1.0/R1)+(1.0/R2))

	Q = w0/b
	Q2 = Q*Q
	v = w*1j/w0
	G = 1.0/np.sqrt((1-u2)**2 + (u2/Q2))
	H = 1.0/(1 + v/Q + v**2)
	#G = G/(1 + R*C*w)
	f0 = w0/(2*np.pi)
    #H = 1.0/(1 + R*C*w)
	return f0,Q,G,H
	

def reject(R1,R2,C1,C2,f):
	w = 2*np.pi*f
	w0 = 1.0/np.sqrt(R1*R2*C1*C2)
	w02 = w0*w0
	u2 = w*w/w02
	
	b = (1.0/C1)*((1.0/R1)+(1.0/R2))

	Q = w0/b
	Q2 = Q*Q
	v = w*1j/w0
	G = (1+u2)/np.sqrt((1-u2)**2 + (u2/Q2))
	H =  (1+v**2)/(1 + v/Q + v**2)
	#G = G/(1 + R*C*w)
	f0 = w0/(2*np.pi)
    #H = 1.0/(1 + R*C*w)
	return f0,Q,G,H	
			
def RC(R,C,f):
	RC2 = R*R*C*C
	w = 2*np.pi*f
	f0 = 1/(2*np.pi*R*C)
	w2=w**2
	v=w*1j
	G = 1.0/np.sqrt(1+RC2*w2)
	H = 1.0/(1 + R*C*v)
	return f0,G,H

p=10**(-12)
n=10**(-9)
k=10**(3)

Ca = [100.0*p,200.0*p,300.0*p,470.0*p]
Rh = 12.0*k
Lh = 6.4
Ch = 80.0*p

Rs = 5.5*k
Ls = 2.2
Cs = (110.0)*p

Rsk = 47.0*k
Csk = 1.0*n

m = 1
n = 8

r1 = Rsk*m
r2 = Rsk/m
c1 = Csk*n
c2 = Csk/n

f = np.linspace(1,20000,20000)
f0sk, Qsk, Gsk, Hsk = sklowpass(r1,r2,c1,c2,f)
f0r, Qr, Gr, Hr = reject(r1,r2,c1,c2,f)
#f0sk, Qsk, Gsk = sallenkey(c1,c2,r1,r2,f)
#f0sk, Qsk, Gsk = sallenkey(Rsk,Rsk,Csk,Csk,f)


#plt.figure()
#plt.grid([200,300])
# Prepare figure #
fig, ax = plt.subplots()
plt.axis([20, 20000, -60, 30])
plt.xscale('log')
plt.grid(True,which="both",ls="-")
plt.xlabel('Freq(Hz)')
plt.ylabel('Gain(dB)')
octfreq = ms.octprint(ax)
#plt.loglog(f,Gh)
#plt.loglog(f,Gs)
#plt.loglog(f,Gs/Gh)

col = ['#ff0000','#00ff00','#0000ff','#00aaaa']

f0,RCf,Hf=RC(500*k,500*p,f)
for i,cai in enumerate(Ca):
	f0h, Qh, Gh, Hh = RLC(Rh,Lh,Ch+cai,f)
	f0s, Qs, Gs, Hs = RLC(Rs,Ls,Cs+cai,f)
	print(i)
	print("f0h = %f, Qh = %f"%(f0h,Qh))
	print("f0s = %f, Qs = %f"%(f0s,Qs))
	print("f0sk = %f, Qsk = %f"%(f0sk,Qsk))

	plt.semilogx(f,db(Gs),c=col[i],label="SC%d"%i)
	plt.semilogx(f,db(Gh),c=col[i],ls="--",label="HB%d"%i)
#plt.semilogx(f,db(Gs/Gh),'g',label="SC/HB")
#plt.semilogx(f,db(RCf*Gs/Gh),'#55ffdd',label="SC/HB")

#plt.semilogx(f,db(RCf),'#55ffdd',label="RC")
#plt.semilogx(f,db(RCf*Gh),'k',label="RC-HB")

#plt.semilogx(f,db(abs(Hs)),'r--',label="SC")
#plt.semilogx(f,db(Gsk),'--',label="SK")
#plt.semilogx(f,db(abs(Hr)),label="Reject")
#plt.semilogx(f,db(abs(Hsk*Hr)),"--",label="+SK+Rej")

#plt.semilogx(f,db(Gs))
#plt.plot(f,Gs/Gh)
plt.legend()
plt.show()
