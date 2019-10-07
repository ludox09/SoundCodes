import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import	FormatStrFormatter
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def octprint(ax):
	""" configurate axes background
	to display musical octaves.
	Also return frequencies of notes
	in tempered scales """
	
    # Define octaves frequencies
	f0 = 55*2**(-9.0/12.0)	
	octfreq = np.zeros((12,10))
	for oct in range(10):
		for note in range(12):
			octfreq[note,oct] = f0*(2)**(note/12.0+oct)
			
	Cfreq = octfreq[0]
	CfreqRound = [int(round(fi)) for fi in Cfreq]
	centfreq = octfreq[4]
	
	# Create octaves block color
	col=["#883333",
	     "#ff5555",
	     "#ffdd00",
	     "#ffff00",
	     "#00cc00",
	     "#00aaff",
	     "#0055ff",
	     "#aaccff",
	     "#ccffff"]
	n = 9
	col = plt.cm.rainbow(np.linspace(0.95,0.05,n))
	c1 = (col[2]+col[3])/2.0
	c2 = (col[5]+col[6])/2.0
	col[2] = c1
	col[3] = c1
	col[5] = c2
	col[6] = c2
	for oct in range(9):
		ax.axvspan(Cfreq[oct],Cfreq[oct+1],alpha=0.7,color=col[oct])
		ax.set_xticks(Cfreq)
		ax.set_xticklabels(CfreqRound, rotation=90)
		
		#ax.set_xticks(Gfreq)
		#ax.set_xticklabels(Gfreq, rotation=90, color="g")
		
		#ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		
	# Create octaves separation
	regions = ["Sub","Bass","Low Mid","","Mid","High Mid","","Presence","Brillance"]
	feelings = ["Chest - Clumsy","Warm - Boomy","Full - Muddy","Boxy","Depth/Body - Honky","Attack - Nasaly/Barky","Sparkenss","Presence - Harsh","Air-Bright/Hiss"]	
	for i in range(9):
		ax.text(centfreq[i],-5,i, size=20, alpha=1, color="k")
		ax.text(centfreq[i],-10,regions[i], rotation=90, size=20, alpha=1, color="k")
		ax.text(centfreq[i],-30,feelings[i], rotation=90, size=12, alpha=1, color="k")
	
			
	#ridx = [0,1,2,4,5,7,8]
	ridx = range(9)
	for i,idx in enumerate(ridx):
		if(idx==3 or idx==6):
			lw = 1
		else:
			lw = 2
		ax.axvline(x=Cfreq[idx], lw= lw, color ="k")
	return octfreq