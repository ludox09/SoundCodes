#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
# Resout equation circuit lineaire
# 12/12/2017

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import matplotlib.image as mpimg
import matplotlib.patches as patches

import mysounds as ms
import wave
import struct
import simpleaudio as sa

# Sound Parameters #
pi = np.pi
fs = 44100
seconds = 2
t = np.linspace(0, seconds, seconds * fs, False)

# Multiples
p=10**(-12)
n=10**(-9)
k=10**(3)

# Pickup physical parameters
Ca = 470.0*p
Rh = 12.0*k
Lh = 6.4
Ch = 80.0*p
Ch = Ch + Ca

Rs = 5.5*k
Ls = 2.2
Cs = (110.0)*p
Cs = Cs + Ca

# Positions parameters
alpha = 0.5
xp = 0.7
xm = 0.7

xNeck_LP = 0.65
xBridge_LP = 0.95

xNeck_TL = 0.75
xBridge_TL = 0.94

x_img_zero_TL = 441.0
x_img_bridge_TL = 2025.0

finalAmp = 0

# Interface blocks
pickuptype = ("Single Coil","Humbucker")
pickupSelector = ("Neck","N+B","N-B","Bridge")
check = [False,False,False]
pick = pickuptype[0]
picksel = pickupSelector[0]

# Prepare figure #
fig = plt.figure()
ax1 = plt.subplot(211)
plt.axis([20, 20000, -60, 20])
plt.xscale('log')
plt.grid(True,which="both",ls="-")
plt.xlabel('Freq(Hz)')
plt.ylabel('Gain(dB)')
octfreq = ms.octprint(ax1)

flist = np.arange(1,10000,100)
curve, = plt.plot(flist,flist,'r',markersize=3)
point, = plt.plot(flist,flist,'ok',markersize=3)
pickresp, = plt.plot(flist,flist,'b')
txt = ax1.text(30,0.5,"f0:", size=10)
plt.subplots_adjust(left=0.25, bottom=0.25)

ax2 = plt.subplot(212)
timg = mpimg.imread("telecaster.jpg")
timg = np.rot90(np.rot90(timg))

xneck = 1900
xbridge = 1935

rect = patches.Rectangle(
        (xneck, 153),
        35,
        35,
        edgecolor = 'black',
        facecolor = '#444444',
        fill=True
     )

ax2.add_artist(rect)



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

playax1 = plt.axes([0.5, 0.9, 0.10, 0.05])
playax2 = plt.axes([0.6, 0.9, 0.10, 0.05])
play1 = Button(playax1, ("Play"))
play2 = Button(playax2, ("Play"))

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
    global finalAmp
    global flist
    global tf

    try:
    	alpha = salpha.val
    except:
    	pass

    try:
    	xp = sxp.val
    except:
    	pass

    try:
        xp = val
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
    	#print("pf =",pf)
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
    	finalAmp = tf*G

    else:
    	flist_rep = np.repeat(flist,3)
    	dbtf = np.repeat(db(tf),3)
    	dbtf[0::3] = -60
    	dbtf[2::3] = -60
    	
    	curve.set_xdata(flist_rep)
    	curve.set_ydata(dbtf)
    	point.set_xdata(flist)
    	point.set_ydata(db(tf))
    	finalAmp = tf

    pickresp.set_xdata(flist)
    pickresp.set_ydata(db(G))
    txt.set_text("f0: %.2f"%(flist[0]))
    fig.canvas.draw_idle() 

def playsound(x):
    global alpha
    global xp
    global xm
    global check
    global pick
    global picksel
    global finalAmp
    global flist
    global tf
 
    flist,tf = Amplitude(alpha, xp)
    pf = PositionFilter(xm)

    #print(flist)
    #print(finalAmp)

    #play.label.set_text("Playing")
    f1 = 110
    t0 = 0.3
    #a1 = np.exp(-t/t0)
    #a1 = 2.0*np.exp(-t/t0)*(t**0.2)

    note = 0
    for amp,fa in zip(finalAmp,flist):
        note = note + np.sin(fa*t*2*np.pi)*amp*np.exp(-fa*t/flist[0])
    note = note/np.max(note) # Renormalize
    
    # Ensure that highest value is in 16-bit range
    audio = note * (2**15 - 1) / np.max(np.abs(note))
    # Convert to 16-bit data
    audio = audio.astype(np.int16)
    
    print("xp == ",xp)
    # Start playback
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    
    #play.label.set_text("Play")
    #fig.canvas.draw_idle() 

def playupdate1(val):
    playsound(1)

def playupdate2(val):
    playsound(2)

def onclick(event):
    global xp
    #global rect
    xdata = event.xdata
    ydata = event.ydata

    if(rect.contains(event)[0]):
        if(rect.get_x() == xneck):
            rect.set_x(xbridge)
        else:
            rect.set_x(xneck)

        fig.canvas.draw() 



    if x_img_zero_TL < xdata and xdata < x_img_bridge_TL and 330 < ydata and ydata < 475:
        #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #  ('double' if event.dblclick else 'single', event.button,
        #   event.x, event.y, event.xdata, event.ydata))

        xp = (xdata - x_img_zero_TL)/(x_img_bridge_TL - x_img_zero_TL)
        update(xp)
        print("xp = ",xp)
        playsound(1)


   
salpha.on_changed(update)
sxp.on_changed(update)
sxm.on_changed(update)
checks.on_clicked(update)
pickup.on_clicked(update)
pickupsel.on_clicked(update)
play1.on_clicked(playupdate1)
play2.on_clicked(playupdate2)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
ax2.imshow(timg)
plt.show()
