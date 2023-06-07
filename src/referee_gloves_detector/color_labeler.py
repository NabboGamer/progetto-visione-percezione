from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2

class ColorLabeler:

	def __init__(self):
		# inizializza il dizionario dei colori, contenenti il nome del colore 
		# come chiave e la tupla RGB come valore
		colors = OrderedDict({
			"red": (255, 0, 0),
			"green": (0, 255, 0),
			"blue": (0, 0, 255)})
		# alloca la memoria per l'immagine L*a*b*, poi inizializza 
		# la lista di nomi dei colori
		self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
		self.colorNames = []
		# itera sul dizionario dei colori
		for (i, (name, rgb)) in enumerate(colors.items()):
			# aggiorna l'array L*a*b* e la lista di nomi dei colori
			self.lab[i] = rgb
			self.colorNames.append(name)
		# converte l'array L*a*b* dallo spazio di colori RGB 
		# allo spazio di colori L*a*b*
		self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

	def label(self, image, c):
		# costruisce una maschera per il contorno, poi calcola
		# il valore medio della L*a*b* per la regione della maschera
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		mask = cv2.erode(mask, None, iterations=2)
		mean = cv2.mean(image, mask=mask)[:3]
		# inizializzare la distanza minima trovata fino a quel momento
		minDist = (np.inf, None)
		# itera sui valori dei colori L*a*b* conosciuti
		for (i, row) in enumerate(self.lab):
			# calcola la distanza tra il valore del colore L*a*b* corrente
			# e il valore medio dell'immagine
			d = dist.euclidean(row[0], mean)
			# se la distanza è minore della distanza corrente,
			# allora aggiorna la variabile contatore
			if d < minDist[0]:
				minDist = (d, i)
		# restituisci il nome del colore con la distanza minore
		return self.colorNames[minDist[1]]