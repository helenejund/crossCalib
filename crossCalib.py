#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Module de calcul de la fonction de transfert d'un sismomètre large-bande par injection d'un bruit blanc (whitecalib)
OU
par comparaison spectrale de capteurs colocalisés

Fonctions:

whiteCalib: Renvoie un tuple de array, définissant la fonction de transfert et les fréquences associées
crossCalib: Renvoie un tuple de array, définissant la fonction de transfert et les fréquences associées. Effectue une déconvolution sur la trace de référence, puis appelle whiteCalib()
Hparameters: Renvoie un tuple de float, définissant la fréquence de coupure, l'amortissement, et la sensibilité

Paramètres:

monitor_trace: Objet trace du module obspy.core.trace correspondant au signal de bruit blanc injecté dans le sismomètre, ou du signal enregistré sur la chaine colocalisée de référence
response_trace: Objet trace du module obspy.core.trace correspondant au signal du sismomètre (réponse de la bobine au bruit blanc)
smooth: Entier définissant le nombre de point utilisés pour moyenner la fonction de transfert calculée (0.01% des points disponibles par défaut)
paz: Dictionnaire poles et zéros (modèle obspy) utilisé pour la déconvolution
fn: Float définissant la fréquence de normalisation (1 Hz par défaut)
plotting: Booléen activant le plot de la fonction de transfert calculée

Executé directement, le fichier propose un exemple de calcul de fonction de transfert d'un T120, par un signal de bruit blanc généré par un Q330

Maxime Bès de Berc
12/11/2013
"""

from obspy.core import read
from obspy.signal import cosTaper
from obspy.signal.util import smooth
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp

def whiteCalib(monitor_trace, response_trace, **kwargs):

	m_trace=monitor_trace.copy()
	r_trace=response_trace.copy()

	#Calcul du spectre du signal moniteur et du spectre de la réponse de la bobine
	mSpectre=np.fft.fft(m_trace.data)
	rSpectre=np.fft.fft(r_trace.data)
	
	#Fréquences associées
	f1=np.fft.fftfreq(len(mSpectre), d=1./m_trace.stats.sampling_rate)
	f2=np.fft.fftfreq(len(rSpectre), d=1./r_trace.stats.sampling_rate)
	
	if len(f1)-len(f2) != 0 or np.sum(f1-f2)!=0:
		print("Warning: frequencies arrays does'nt have same length or frequencies arrays not strictly identical, mixing")
		fmax=np.minimum(np.amax(np.absolute(f1)), np.amax(np.absolute(f2)))
		i1=np.where(np.absolute(f1)<fmax)
		f1=f1[i1]
		mSpectre=mSpectre[i1]
		i2=np.where(np.absolute(f2)<fmax)
		f2=f2[i2]
		rSpectre=rSpectre[i2]

		if len(f1)>len(f2):
			i=np.argmax(f1)
			f1=np.delete(f1, i)
			mSpectre=np.delete(mSpectre, i)
			i=np.argmin(f1)
			f1=np.delete(f1, i)
			mSpectre=np.delete(mSpectre, i)
		elif len(f2)>len(f1):
			i=np.argmax(f2)
			f2=np.delete(f2, i)
			rSpectre=np.delete(rSpectre, i)
			i=np.argmin(f2)
			f2=np.delete(f2, i)
			rSpectre=np.delete(rSpectre, i)			
		
	#Division spectrale brute (calcul de la fonction de transfert)
	H=rSpectre/mSpectre

	if 'smooth' in kwargs:
		H=smooth(H, kwargs['smooth'])
	else:
		H=smooth(H, int(len(f1)*0.0001))
	
	return (H,f1) 

def crossCalib(monitor_trace, response_trace, **kwargs):
	
	in_trace=monitor_trace.copy()
	out_trace=response_trace.copy()
	
	if 'ffilter' in kwargs:
		in_trace.filter('highpass', freq=kwargs['ffilter'], corners=4, zerophase=True)
		out_trace.filter('highpass', freq=kwargs['ffilter'], corners=4, zerophase=True)

	if 'deconvolve' in kwargs and kwargs['deconvolve']:
		if 'paz' in kwargs:
			in_trace.simulate(paz_remove=kwargs['paz'], taper=False, zero_mean=False)
		else:
			print("Error: No paz for deconvolution")
		
	if 'smooth' in kwargs:
		(H,f)=whiteCalib(in_trace, out_trace, smooth=kwargs['smooth'])
	else:
		(H,f)=whiteCalib(in_trace, out_trace)

	return (H,f)

def Hparameters(H, f, **kwargs):
	
	#Periode propre determinée lorsque lorsque H est strictement imaginaire, ie Phi=+/-90°
	#On travaille dans les fréquences inférieures à la fréquence de normalisation (souvent fn=1Hz pour les LB)
	# et à des fréquences supérieures à fmin. Celle-ci est à determiner au cas par cas en fonction des premiers résultats.
	
	#Détermination de l'indice de fnorm
	if 'fnorm' in kwargs:
		inorm=int(np.absolute(f-kwargs['fnorm']).argmin())
	else:
		inorm=int(np.absolute(f-1).argmin())

	#Détermination de l'indice de fmin
	if 'fmin' in kwargs:
		imin=int(np.absolute(f-kwargs['fmin']).argmin())
	else:
		imin=int(np.absolute(f-0.001).argmin())

	#Détermination de l'indice de la fréquence de coupure
	indx=int(np.absolute(np.real(H[imin:inorm])).argmin())
	indx=indx+imin

	#Fonction de transfert en amplitude normalisée et phase
	#Interpolation 1d du bruit lors des phases de mouvement de la table
	Amp=np.absolute(H)/np.absolute(H[inorm])
	Phi=np.angle(H, deg=True)
	
	if 'plotting' in kwargs and kwargs['plotting']:
		plt.figure()
		plt.subplot(211)
		plt.semilogx(f[1:], 20*np.log10(Amp[1:]), 'b-')
		plt.xlim([f[imin], np.amax(f)])
		plt.grid()
		plt.ylabel("Amplitude (dB)")
		plt.subplot(212)
		plt.semilogx(f[1:], Phi[1:], 'r-')
		plt.xlim([f[imin], np.amax(f)])
		plt.grid()
		plt.ylabel("Phase (deg)")
		plt.xlabel("Frequence (Hz)")
		plt.show()
	
	return (f[indx], 1./(2*(Amp[indx])), np.absolute(H[inorm]) )

if __name__=='__main__':
	print("Calcul de la fonction de transfert d'un T120 sur un Q330")
	m_trace=read("test/monitor.B1.BHZ.seed")[0]
	r_trace=read("test/response.A1.BHZ.seed")[0]
	r_trace.data=r_trace.data*(-1)
	
	m_trace.detrend('demean')
	m_trace.taper()
	r_trace.detrend('demean')
	r_trace.taper()

	(H,f)=whiteCalib(m_trace, r_trace)
	H=H*2j*np.pi*f
	print(Hparameters(H, f, plotting=True))
