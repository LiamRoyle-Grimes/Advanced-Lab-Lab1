# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 23:55:52 2025

@author: liamr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

def poisson(x,mu):
    return(mu**x*np.e**-mu)/scipy.special.factorial(x)

def linear(x,m,b):
    return(m*x+b)

def plateu(x,a):
    return(0*x+a)
def invsqr(x,a,b,c):
    return a+b*(x-c)**-2
i,v,c,t=np.loadtxt("plateau data.tsv", skiprows =11,usecols=(0,1,2,3),unpack=True)
def exponential(x,a,b,c):
    return a*np.exp(b*x)+c

GRAPH=True



lin,covlin=scipy.optimize.curve_fit(linear,v,c)


plat,covplat=scipy.optimize.curve_fit(plateu,v,c)
#print(plateu(v,plat[0]))


Background=pd.read_csv("background.tsv",sep="\t",skiprows=9,usecols=[1,2,3,4])

#print(Background.to_latex())


#print(np.mean(Background["Counts"]),np.std(Background["Counts"]))

BGglobal=np.mean(Background["Counts"])/300
errBGglobal=np.std(Background["Counts"])/300

Beta5min=pd.read_csv("beta5min.tsv",sep="\t",skiprows=9,usecols=[1,2,3,4])

print(Beta5min.to_latex())
print(np.mean(Beta5min["Counts"]),np.std(Beta5min["Counts"]))
print(np.mean(Beta5min["Counts"])/300,np.std(Beta5min["Counts"])/300)

BetaEmission=np.mean(Beta5min["Counts"])/300

PoissonAlpha=pd.read_csv("poisson alpha.tsv",sep="\t",skiprows=10,usecols=[1,2,3,4])
PoissonBackground=pd.read_csv("poisson background.tsv",sep="\t",skiprows=10,usecols=[1,2,3,4])
#print(np.mean(PoissonBackground["Counts"]))

#print(np.std(PoissonBackground["Counts"]))

#print(np.sqrt(np.mean(PoissonBackground["Counts"])))

#print("alpha present")

#print(np.mean(PoissonAlpha["Counts"]))

#print(np.std(PoissonAlpha["Counts"]))

#print(np.sqrt(np.mean(PoissonAlpha["Counts"])))


BetaShield=pd.read_csv("beta shield.tsv",sep="\t",skiprows=10,usecols=[1,2,3,4])
AlphaShield=pd.read_csv("alph shield2.tsv",sep="\t",skiprows=10,usecols=[1,2,3,4])

BetaDistance=pd.read_csv("beta dist.tsv",sep="\t",skiprows=10,usecols=[1,2,3,4])
BetaDistBKG=pd.read_csv("beta dist background.tsv",sep="\t",skiprows=10,usecols=[1,2,3,4])
BG=(np.mean(BetaDistBKG["Counts"]),np.sqrt(np.mean(BetaDistBKG["Counts"])))
#print(f" Background ={BG}")
BetaShelves=np.array([1+ i//2 for i in range(18)])

invsqrmodel,cov=scipy.optimize.curve_fit(invsqr,BetaShelves,BetaDistance["Counts"],p0=[BG[0],1400,0])
#print(invsqrmodel)


shields=np.array(['T','S','R','Q','P','O','M','N','L','K','J','I','H','G','F','E','D','C','B','A'])
materials=np.array(np.concatenate([['Pb'for i in range(4)],['Al' for i in range(10)],["Plastic" for i in range(2)],["Poly" for i in range(2)],["Al" for i in range(2)]]))
thicknesses=np.array([0.250,0.125,0.065,0.032,0.125,0.100,0.090,0.080,0.063,0.050,0.040,0.032,0.025,0.020,0.040,0.030,0.008,0.004,0.001,0.0007])
densities=np.array([7367,3348,2066,1120,840,655,645,522,425,328,258,216,170,141,102,59.1,19.2,9.6,6.5,4.5])

shields=np.stack([shields,materials,thicknesses,densities],axis=1)

Shields=pd.DataFrame(shields,columns=["shields","materials","thicknesses","densities"])
#print(Shields)
#print(Shields.to_latex())


if GRAPH:
    fig1,ax1=plt.subplots(2,sharex=True)
    ax1[0].plot(v,linear(v,lin[0],lin[1]),color="blue",linestyle="dotted",label="model",marker="none")
    ax1[0].errorbar(v,c,yerr=np.sqrt(c),marker="o",linestyle="none",color="red", label="data")
    ax1[1].plot(v,(linear(v,lin[0],lin[1])-c)**2,marker=".",linestyle="none",label="residuals$^2$")
    ax1[1].set_xlabel("Voltage Volts")
    ax1[0].set_ylabel("Counts")
    ax1[1].set_ylabel("Squared Residuals")
    fig1.legend()
    fig1.show()
    
    fig2,ax2=plt.subplots(2,sharex=True)
    ax2[0].plot(v,plateu(v,plat),color="Purple",linestyle="dotted",label="model",marker="none")
    ax2[0].errorbar(v,c,yerr=np.sqrt(c),marker="o",linestyle="none",color="red", label="data")
    ax2[1].plot(v,(plateu(v,plat)-c)**2,marker=".",linestyle="none",label="residuals$^2$")
    ax2[1].set_xlabel("Voltage (V)")
    ax2[0].set_ylabel("# of Detections")
    ax2[1].set_ylabel("Squared Residuals")
    fig2.legend()
    fig2.show()
    
    fig3,ax3=plt.subplots()
    ax3.hist(PoissonAlpha["Counts"],bins=14,alpha=0.5,label="Alpha source present")
    ax3.hist(PoissonBackground["Counts"],bins=14,alpha=0.5,label="Background")
    ax3.set_xlabel("# of Detections")
    ax3.set_ylabel("Count")
    fig3.legend()
    fig3.show()
    
    fig4,ax4=plt.subplots()
    ax4.errorbar(BetaShelves,BetaDistance["Counts"], yerr=np.sqrt(BetaDistance["Counts"]),linestyle="none",marker=".",label="Data")
    a,b,c=invsqrmodel
    ax4.plot(BetaShelves,invsqr(BetaShelves,a,b,c),label="Model")
    ax4.set_xlabel("Distance (shelves)")
    ax4.set_ylabel("# of Detections")
    fig4.legend()
    print(scipy.stats.chisquare(BetaDistance["Counts"],invsqr(BetaShelves,a,b,c),15))
    
    
    fig5,ax5=plt.subplots()
    
    A=BetaShield["Counts"][:8]
    B=BetaShield["Counts"][8:26]
    
    
    ThA=np.array([float(Shields["thicknesses"][i//2]) for i in range(8)])
    ThB=np.array([float(Shields["thicknesses"][i//2]) for i in range(8,26)])
    
    mask=np.ones(A.shape,dtype=bool)
    mask[4]=0
    
    ax5.errorbar(ThA,A,yerr=np.sqrt(A),linestyle="none",marker=".",label="Pb")
    ax5.errorbar(ThB,B,yerr=np.sqrt(B),linestyle="none",marker=".",label="Al")
    
    param,cov=scipy.optimize.curve_fit(exponential,ThA[mask],A[mask],p0=[732,-1.5,0])
    ax5.plot(ThA,exponential(ThA,param[0],param[1],param[2]),label=f"${param[0]:0.0f}e^{{{param[1]:0.1f}x}}+{param[2]:0.0f}$")
    
    param,cov=scipy.optimize.curve_fit(exponential,ThB,B,p0=[732,-1.5,0])
    ax5.plot(ThB,exponential(ThB,param[0],param[1],param[2]),label=f"${param[0]:0.0f}e^{{{param[1]:0.1f}x}}+{param[2]:0.0f}$")
    
    ax5.set_xlabel("Barier thickness (in)")
    ax5.set_ylabel("# of Detections")
    fig5.legend()
    fig5.show()
    
    fig6,ax6=plt.subplots()
    BGlocal= np.mean(AlphaShield["Counts"][-2:])
    print(poisson(np.arange(40,80,step=4), BGglobal))
    
    results, edges = np.histogram(AlphaShield["Counts"][3:],bins=np.arange(40,80,step=4),density=True)
    width= edges[1]-edges[0]
    ax6.bar(edges[:-1],results,width=width,label="# of detections")
    model,cov=scipy.optimize.curve_fit(poisson,edges[:-1]-width/2,results,p0=np.mean(AlphaShield["Counts"][3:]))
    
    ax6.plot(edges[:]-width/2,poisson(edges[:]-width/2,model[0]),color="red")
    ax6.errorbar(model[0],poisson(model[0],model[0]),xerr=np.sqrt(model[0]), marker="s",capsize=10,color="red",label="best fit model")
    
    ax6.plot(edges[:]-width/2,poisson(edges[:]-width/2, BGlocal),color="black",label="Background$_{new}$")
    ax6.axvline(BGlocal,color="black")
    
    ax6.plot(edges[:]-width/2,poisson(edges[:]-width/2, BGglobal*60),color="purple",label="Background$_{old}$")
    ax6.axvline(BGglobal*60,color="purple")
    
    ax6.set_xlabel("# of Detections")
    ax6.set_ylabel("Frequency")
    
    fig6.legend()
    fig6.show()
    
    fig1.savefig("figure 1")
    fig2.savefig("figure 2")
    fig3.savefig("figure 3")
    fig4.savefig("figure 4")
    fig5.savefig("figure 5")
    fig6.savefig("figure 6")
    
