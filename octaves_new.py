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
alpha = 0
xp = 0.7
xm = 0.7
nmax = 90

xNeck_LP = 0.65
xBridge_LP = 0.95

xNeck_TL = 0.75
xBridge_TL = 0.94

xNeck   = {"Single Coil":xNeck_TL, "Humbucker":xNeck_LP}
xBridge = {"Single Coil":xBridge_TL, "Humbucker":xBridge_LP}


x_img_zero_TL = 441.0
x_img_bridge_TL = 2025.0

finalAmp = 0

# Interface blocks

# Prepare figure #
fig = plt.figure()
ax1 = plt.subplot(211)
plt.axis([20, 20000, -60, 20])
plt.xscale('log')
plt.grid(True,which="both",ls="-")
plt.xlabel('Freq(Hz)')
plt.ylabel('Gain(dB)')
octfreq = ms.octprint(ax1)
fstring=[octfreq[4,1], # E
         octfreq[9,1], # A
         octfreq[2,2], # D
         octfreq[7,2], # G
         octfreq[11,2], # B
         octfreq[4,3]] # E

flist = np.arange(1,10000,100)
curve, = plt.plot(flist,flist,'r',markersize=3,linewidth=0.7)
point, = plt.plot(flist,flist,'ok',markersize=2)
decay1, = plt.plot(flist,flist,'or',markersize=2)
decay2, = plt.plot(flist,flist,'og',markersize=2)
decay3, = plt.plot(flist,flist,'ob',markersize=2)
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
salpha = Slider(axalpha, 'alpha', 0, 5, valinit=1, valstep=1,valfmt="%d")

axxm = plt.axes([0.25, 0.05, 0.65, 0.02])
sxm = Slider(axxm, 'xm', 0.001, 0.99, valinit=0.5)

checkax = plt.axes([0.05, 0.7, 0.15, 0.3])
checks = CheckButtons(checkax, ('Pickup FR', 'xp','xm'), (False, False, False))
check=[False,False,False]

pickuptype = ("Single Coil","Humbucker")
pickuptypeax = plt.axes([0.05, 0.4, 0.15, 0.3])
pickup = RadioButtons(pickuptypeax, pickuptype)
pick=pickuptype[0]

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
        f0 = fstring[int(alpha)]
        tf = []
        flist =[]
        for n in range(1,nmax):
            flist.append(n*f0)
            #tf.append(1.0/n)
            #tf.append((2.0/(xp*(1-xp)))*np.sin(np.pi*n*xp)/float(n**2))
            tf.append(np.sin(np.pi*n*xp)/float(n**2))
        flist = np.array(flist)
        tf = np.array(tf)
        return flist,tf

def PositionFilter(xm):
	pf = []
	for n in range(1,nmax):
		#tf.append(1.0/n)
		pf.append(np.sin(np.pi*n*xm))
	pf = np.array(pf)
	return pf

def update_alpha(val):
    global alpha
    alpha = salpha.val
    update()

#def update_xm():
#    global xp
#    xp = sxp.val

def update_xm():
    global xm
    xm = sxm.val
    update()

def update_checks(val):
    global check
    check = [checks.lines[0][0].get_visible(),checks.lines[1][0].get_visible(),checks.lines[2][0].get_visible()]
    update()

def update_pickup(val):
    global pick
    if val in pickuptype:
    	pick = val
    update()
 
def update():
    global alpha
    global xp
    global xm
    global check
    global checks
    global pick
    global picksel
    global finalAmp
    global flist
    global tf
   
   	
    flist,tf = Amplitude(alpha, xp)
    pf = PositionFilter(xm)

    # Activate checked module
    if (check[1],check[2])==(False,False):
    	tf = tf*0.0+1.0
    elif (check[1],check[2])==(False,True):
    	tf = pf
    	#print("pf =",pf)
    elif (check[1],check[2])==(True,False):
    	pass
    elif (check[1],check[2])==(True,True):
    	tf = tf*pf
     
    # Pickup frequency response filter 
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

def fdecay(f,t):
    return np.exp(-f*t/f[0])

def playsound(mode=0):
    global alpha
    global xp
    global xm
    global check
    global pick
    global picksel
    global finalAmp
    global flist
    global tf

    if mode == 0:
        flist,tf = Amplitude(alpha, xp)
        pf = PositionFilter(xm)

        t1 = 0.1
        t2 = 0.2
        t3 = 0.3
        #a1 = np.exp(-t/t0)
        #a1 = 2.0*np.exp(-t/t0)*(t**0.2)

        #note = 0
        #for amp,fa in zip(finalAmp,flist):
        #    note = note + np.sin(fa*t*2*np.pi)*amp*np.exp(-fa*t/flist[0])
        tt,ff = np.meshgrid(t,flist)

        notes = np.sin(2.0*np.pi*ff*tt)*fdecay(ff,tt)
        note = np.sum(notes*finalAmp[:,None],axis=0)
        note = note/np.max(note) # Renormalize
        
        # Ensure that highest value is in 16-bit range
        audio = note * (2**15 - 1) / np.max(np.abs(note))
        # Convert to 16-bit data
        audio = audio.astype(np.int16)
        
        #print("xp == ",xp)
        # Start playback

        print(6)
        play_obj = sa.play_buffer(audio, 1, 2, fs)
        decay1.set_xdata(flist)
        decay2.set_xdata(flist)
        decay3.set_xdata(flist)
        decay1.set_ydata(db(finalAmp*fdecay(flist,t1)))
        decay2.set_ydata(db(finalAmp*fdecay(flist,t2)))
        decay3.set_ydata(db(finalAmp*fdecay(flist,t3)))
        
        #play.label.set_text("Play")
        #fig.canvas.draw_idle()

    if mode == 1:

        t0 = 0.3
        note = 0
        for alpha in range(6):
            flist,tf = Amplitude(alpha, xp)
            pf = PositionFilter(xm)

            for amp,fa in zip(finalAmp,flist):
                note = note + np.sin(fa*t*2*np.pi)*amp*np.exp(-fa*t/flist[0])
       
        note = note/np.max(note) # Renormalize
         
        # Ensure that highest value is in 16-bit range
        audio = note * (2**15 - 1) / np.max(np.abs(note))
        # Convert to 16-bit data
        audio = audio.astype(np.int16)
        
        # Start playback
        play_obj = sa.play_buffer(audio, 1, 4, fs)
        
        #play.label.set_text("Play")
        #fig.canvas.draw_idle()


def playupdate1(val):
    playsound(1)

def playupdate2(val):
    playsound(2)

def onclick(event):
    global xp
    global xm
    global pick

    #global rect
    xdata = event.xdata
    ydata = event.ydata

    if(rect.contains(event)[0]):
        if(rect.get_x() == xneck):
            rect.set_x(xbridge)
            xm = xBridge[pick]



        else:
            rect.set_x(xneck)
            xm = xNeck[pick]
        update()
        fig.canvas.draw() 

    if x_img_zero_TL < xdata and xdata < x_img_bridge_TL and 330 < ydata and ydata < 475:
        #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #  ('double' if event.dblclick else 'single', event.button,
        #   event.x, event.y, event.xdata, event.ydata))
        xp = (xdata - x_img_zero_TL)/(x_img_bridge_TL - x_img_zero_TL)
        update()
        #print("xp = ",xp)
        playsound()

   
salpha.on_changed(update_alpha)
sxm.on_changed(update_xm)
checks.on_clicked(update_checks)
pickup.on_clicked(update_pickup)

play1.on_clicked(playupdate1)
play2.on_clicked(playupdate2)

cid = fig.canvas.mpl_connect('button_press_event', onclick)
ax2.imshow(timg)
plt.show()
