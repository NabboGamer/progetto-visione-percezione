import sys
sys.path.insert(19, '../../litepose-pose-estimation/src')
import json
import torch
from torchvision import transforms
import lp_config.lp_common_config as cc
from lp_model.lp_litepose import LitePose
from lp_inference.lp_inference import inference, assocEmbedding
from lp_utils.lp_image_processing import drawHeatmap, drawKeypoints, drawSkeleton
import cv2
from PIL import Image
from datetime import datetime
from preprocessing import red_filtering, segmentation_and_cropping, equalizing, squaring


with open('../../config/config.json') as f:
    config_data = json.load(f)
# Ottengo i percorsi dei file
file_path_big_arch = config_data['path_big_arch']
file_path_csv_keypoints_webcam = config_data['path_csv_keypoints_webcam']


model = LitePose().to(cc.config["device"])
model.load_state_dict(torch.load(file_path_big_arch, map_location=cc.config["device"]))


webcam=cv2.VideoCapture(0) 
if not webcam.isOpened():
    raise Exception("Errore nell'apertura della webcam")
keypoints_vec = []
timestamps = []
resize = transforms.Resize([224, 224])  
to_tensor = transforms.ToTensor()
ret,frame=webcam.read()
while ret:
    istante_attuale = datetime.now()
    stringa_istante = istante_attuale.strftime("%d/%m/%Y %H:%M:%S.%f")
    timestamps.append(stringa_istante)   
    frame = cv2.flip(frame, 1)
    #-----------------------------PREPROCESSING START HERE-----------------------------
    #RED FILTERING
    full_mask = red_filtering(frame)
    #SEGMENTATION E CROPPING
    cropped_image = segmentation_and_cropping(frame, full_mask)
    #NORMALIZATION
    normalized_image = cv2.normalize(cropped_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #EQUALIZATION
    equalized_image = equalizing(normalized_image)
    #------------------------------PREPROCESSING END HERE------------------------------
    img = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)
    img = squaring(img)
    try:
        # Può dare problemi se gli arriva una immagine con una dimensione mancante,
        # non ho capito il perchè (anche perchè è una condizione difficilmente riproducibile) 
        # ma questo può succedere
        im_pil = Image.fromarray(img)
    except:
        print("Shape dell'immagine che fa crashare il processo: " + str(img.shape))
        pass
    frame = resize(im_pil)
    tensor = to_tensor(frame)
    tensor = tensor.unsqueeze(0)
    if ret==True:
        output, keypoints = inference(model, tensor)
        keypoints_vec.append(keypoints)
        embedding = assocEmbedding(keypoints)
        frame_modified = drawSkeleton(tensor[0], embedding[0])
        #frame = drawKeypoints(tensor[0], keypoints[0])
        cv2.imshow("Pose estimation", frame_modified)
        key=cv2.waitKey(1) & 0xFF
        if key==ord("q"):
            break         
    ret, frame = webcam.read()
webcam.release()
cv2.destroyAllWindows()
print("Dimensioni dell'array di frame:", len(keypoints_vec))
print("Dimensioni dell'array di timestamp:", len(timestamps))