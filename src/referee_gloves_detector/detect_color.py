## per lanciare il codice aprire il terminale e scrivere "$ python detect_color.py --image example_shapes.png"

# importa i package necessari
from color_labeler import ColorLabeler
import argparse
import imutils
import cv2
# costruisce il parser di argomenti e analizza gli argomenti
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# carica l'immagine ed effettua il resize ad una scala minore affinché le shape possono essere approssimate meglio
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])
# aggiunge del leggero Blur all'immagine, dopodiché la converte in Scala di grigi e spazio di colori L*a*b* 
blurred = cv2.GaussianBlur(resized, (5, 5), 0)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB) 
thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
# trova i contorni nell'immagine con soglia
# controllo se sto usando OpenCV 2.X o OpenCV 4
if imutils.is_cv2() or imutils.is_cv4():
  (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
# controllo se sto usando OpenCV 3
elif imutils.is_cv3():
  (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
# inizializza il rilevatore di forma e l'etichettatore di colore 
# sd = ShapeDetector()
cl = ColorLabeler()


# itera sui contorni
for c in cnts:
	# calcola il centro del contorno
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	# rileva la forma del contorno ed etichetta il colore
	# shape = sd.detect(c)
	color = cl.label(lab, c)
	print(color)
	print(type(color))
	if(color != "red"):continue
	# moltiplica le coordinate-(x,y) del contorno per il rapporto d'aspetto ridimensionato,
	# poi disegna i contorni e il nome della forma assieme al colore etichettato sull'immagine 
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	text = "{} {}".format(color, "glove")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, text, (cX, cY),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	# mostra l'immagine finale
	cv2.imshow("Image", image)
	cv2.waitKey(0)