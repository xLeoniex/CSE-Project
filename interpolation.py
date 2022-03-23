import time
import cv2
import numpy as np
from numpy import asarray, ndarray
import os
from cv2 import mean

# alle benötigten Listen
# list_height = []
# list_width = []
list_time_lin = []
list_time_near = []

# Pfad und filesliste für ampera(30,23), mit amz,meisten bilder(46,37)
input_path_1 = "C:\\Users\\leoni\\Desktop\\Uni\\CSE_projekt\\fsoco_bounding_boxes_train\\ampera\\output\\"
files = os.listdir(input_path_1)

#anzahl_Bilder = 0
# # höhe und breite berechnen
# for file in files:
#     path = input_path_1 + file
#     img = cv2.imread(path)
#     height, width, _ = img.shape
#     list_height.append(height)
#     list_width.append(width)
# mean_height_1 = np.mean(list_height)
# mean_width_1 = np.mean(list_width)
# #mean_height = int(mean_height_1)
#mean_width = int(mean_width_1)

mean_height = 26
mean_width = 26

# Pfad um durch fsoco_bounding_boxes durchzulaufen
big_path = "C:\\Users\\leoni\\Desktop\\Uni\\CSE_projekt\\fsoco_bounding_boxes_train\\"
files_big = os.listdir(big_path)
files_big.remove('meta.json')

for big in files_big:
    print(big)
    Path_NewFolder_l = big_path + big + "\\linear\\"
    Path_NewFolder_n = big_path + big + "\\nearest\\"
    if not os.path.exists(Path_NewFolder_l):
         os.makedirs(Path_NewFolder_l)
    if not os.path.exists(Path_NewFolder_n):
        os.makedirs(Path_NewFolder_n)
    # input_path -> Bilder der Hüttchen in verschiedenen Größen gespeichert
    # Save_path -> abspeichern der fertig interpolierten Bilder
    input_path = "C:\\Users\\leoni\\Desktop\\Uni\\CSE_projekt\\fsoco_bounding_boxes_train\\" + big + "\\output\\"
    save_path_lin = "C:\\Users\\leoni\\Desktop\\Uni\\CSE_projekt\\fsoco_bounding_boxes_train\\" + big + "\\linear\\"
    save_path_near = "C:\\Users\\leoni\\Desktop\\Uni\\CSE_projekt\\fsoco_bounding_boxes_train\\" + big + "\\nearest\\"
    # durch die Ordner output durchlaufen
    files = os.listdir(input_path)

    start_time_lin: float = time.time()
    for file in files:
        # Bild in array umwandeln
        # f = open(input_path + file, 'rb')
        #
        #
        # image_bytes = f.read()
        # data = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        # f.close()
        #anzahl_Bilder = anzahl_Bilder+1
        data = cv2.imread(input_path + file)

        # lineare interpolation und im Ordner linear abspeichern
        lin = cv2.resize(data, dsize=(mean_width, mean_height), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(save_path_lin + file, lin)
    # Zeit für die lineare Interpolation
    Time_lin = time.time() - start_time_lin
    list_time_lin.append(str(Time_lin))
    print("Time linear:" + str(Time_lin))

    start_time_near = time.time()
    for file in files:
        # Bild in array umwandeln
        # f = open(input_path + file, 'rb')
        # image_bytes = f.read()
        # data = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        data = cv2.imread(input_path + file)

        # nearest-Neighbour interpolation und im Ordner nearest abspeichern
        near = cv2.resize(data, dsize=(mean_width, mean_height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_path_near + file, near)

    # Zeit für die nearest-Neighbour-Interpolation
    Time_near = time.time() - start_time_near
    list_time_near.append(Time_near)
    print("Time nearest :" + str(Time_near))
    #print(anzahl_Bilder)