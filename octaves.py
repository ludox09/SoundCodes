#!/usr/bin/python
# -*- coding: utf-8 -*-
# Resout equation circuit lineaire
# 12/12/2017

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import mysounds as ms
import wave
import struct

# Parameter #
frate = 44100.0  # framerate as a float

p=10**(-12)
n=10**(-9)
k=10**(3)

Ca = 470.0*p
Rh = 12.0*k
Lh = 6.4
Ch = 80.0*p
Ch = Ch + Ca

Rs = 5.5*k
Ls = 2.2
Cs = (110.0)*p
Cs = Cs + Ca

pi = 3.14159265359

alpha = 0.5
xp = 0.7
xm = 0.7

xNeck_LP = 0.65
xBridge_LP = 0.95

xNeck_TL = 0.75
xBridge_TL = 0.94

pickuptype = ("Single Coil","Humbucker")
pickupSelector = ("Neck","N+B","N-B","Bridge")
check = [False,False,False]
pick = pickuptype[0]
picksel = pickupSelector[0]

# Prepare figure #
fig, ax = plt.subplots()
plt.axis([20, 20000, -60, 20])
plt.xscale('log')
plt.grid(True,which="both",ls="-")
plt.xlabel('Freq(Hz)')
plt.ylabel('Gain(dB)')
octfreq = ms.octprint(ax)


flist = np.arange(1,10000,100)
curve, = plt.plot(flist,flist,'r',markersize=3)
point, = plt.plot(flist,flist,'ok',markersize=3)
pickresp, = plt.plot(flist,flist,'b')
txt = ax.text(30,0.5,"f0:", size=10)
plt.subplots_adjust(left=0.25, bottom=0.25)

# Widget
axalpha = plt.axes([0.25, 0.11, 0.65, 0.02])
salpha = Slider(axalpha, 'alpha', 0.001, 0.99, valinit=0.5)

axxp = plt.axes([0.25, 0.08, 0.65, 0.02])
sxp = Slider(axxp, 'xp', 0.001, 0.99, valinit=0.5)

axxm = plt.axes([0.25, 0.05, 0.65, 0.02])
sxm = Slider(axxm, 'xm', 0.001, 0.99, valinit=0.5)

checkax = plt.axes([0.05, 0.7, 0.15, 0.3])
checks = CheckButtons(checkax, ('Pickup FR', 'xp','xm'), (False, False, False))

pickuptypeax = plt.axes([0.05, 0.4, 0.15, 0.3])
pickup = RadioButtons(pickuptypeax, pickuptype)

pickupselax = plt.axes([0.05, 0.1, 0.15, 0.3])
pickupsel = RadioButtons(pickupselax, pickupSelector)

exportax = plt.axes([0.5, 0.9, 0.15, 0.05])
export = Button(exportax, ("Export"))

def RLC(R,L,C,f):
	""" Calculate and output
	- Cut-off frequency
	- Q factor
	- Gain
	of a RLC low-pass filter
	"""
	RC2 = R*R*C*C
	w0 = 1.0/np.sqrt(L*C)
	w02 = w0*w0
	Q = w0*L/R
	Q2 = Q*Q
	w = 2*np.pi*f
	u2 = w*w/w02
	G = 1.0/np.sqrt((1-u2)**2 + (u2/Q2))
	f0 = w0/(2*np.pi)
	return f0,Q,G
	
def RC(R,C,f):
	""" Calculate and output
	the gain of a RC low-pass filter """
	
	RC2 = R*R*C*C
	w = 2*np.pi*f
	w2=w**2
	G = 1.0/np.sqrt(1+RC2*w2)
	return G
    
def db(x):
	""" Return value in decibel"""
	return 10.0*np.log10(np.abs(x))
	
def Amplitude(alpha,xp):
	f0=octfreq[4,1]*2**(alpha)
	tf = []
	flist =[]
	for n in range(1,150):
		flist.append(n*f0)
		#tf.append(1.0/n)
		#tf.append((2.0/(xp*(1-xp)))*np.sin(np.pi*n*xp)/float(n**2))
		tf.append(np.sin(np.pi*n*xp)/float(n**2))
	flist = np.array(flist)
	tf = np.array(tf)
	return flist,tf

def PositionFilter(xm):
	pf = []
	for n in range(1,150):
		#tf.append(1.0/n)
		pf.append(np.sin(np.pi*n*xm))
	pf = np.array(pf)
	return pf


def update(val):
    global alpha
    global xp
    global xm
    global check
    global pick
    global picksel

    try:
    	alpha = salpha.val
    except:
    	pass

    try:
    	xp = sxp.val
    except:
    	pass

    try:
    	xm = sxm.val
    except:
    	pass

    try:
    	check = [checks.lines[0][0].get_visible(),checks.lines[1][0].get_visible(),checks.lines[2][0].get_visible()]
    except:
    	pass

    if val in pickuptype:
    	pick = val
 
    if val in pickupSelector:
    	picksel = val
    	
    print(alpha, check, pick,xp,xm)
    if (pick==pickuptype[0]):
       # Single Coil #
       if (picksel==pickupSelector[0]):
           xm = xNeck_TL
       elif (picksel==pickupSelector[3]):
           xm = xBridge_TL
    elif (pick==pickuptype[1]):
       # Humbucker #
       if (picksel==pickupSelector[0]):
           xm = xNeck_LP
       elif (picksel==pickupSelector[3]):
           xm = xBridge_LP
   	
    flist,tf = Amplitude(alpha, xp)
    pf = PositionFilter(xm)
    if (check[1],check[2])==(False,False):
    	tf = tf*0.0+1.0
    elif (check[1],check[2])==(False,True):
    	tf = pf
    	print("pf =",pf)
    elif (check[1],check[2])==(True,False):
    	pass
    elif (check[1],check[2])==(True,True):
    	tf = tf*pf
     
    
    if pick == pickuptype[0]:
    	f0, Q, G = RLC(Rs,Ls,Cs,flist)
    if pick == pickuptype[1]:
    	f0, Q, G = RLC(Rh,Lh,Ch,flist)
    		
    				
    if check[0]:
    	flist_rep = np.repeat(flist,3)
    	dbtf = np.repeat(db(tf*G),3)
    	dbtf[0::3] = -60
    	dbtf[2::3] = -60
    	
    	curve.set_xdata(flist_rep)
    	curve.set_ydata(dbtf)
    	point.set_xdata(flist)
    	point.set_ydata(db(tf*G))
    else:   	
    	flist_rep = np.repeat(flist,3)
    	dbtf = np.repeat(db(tf),3)
    	dbtf[0::3] = -60
    	dbtf[2::3] = -60
    	
    	curve.set_xdata(flist_rep)
    	curve.set_ydata(dbtf)
    	point.set_xdata(flist)
    	point.set_ydata(db(tf))
    pickresp.set_xdata(flist)
    pickresp.set_ydata(db(G))
    txt.set_text("f0: %.2f"%(flist[0]))
    fig.canvas.draw_idle() 

def exportupdate(val):
    #global alpha
    txt.set_text("Export")
    fig.canvas.draw_idle() 

salpha.on_changed(update)
sxp.on_changed(update)
sxm.on_changed(update)
checks.on_clicked(update)
pickup.on_clicked(update)
pickupsel.on_clicked(update)
export.on_clicked(exportupdate)
plt.show()

