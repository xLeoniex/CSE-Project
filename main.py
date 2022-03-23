import json
import logging
import os
from os import listdir
from os.path import isfile, join
import cv2

#durch fsoco_bounding_boxes durchlaufen(ampera,amz,...)
big_path = "C:\\Users\\leoni\\Desktop\\Uni\\CSE_projekt\\fsoco_bounding_boxes_train\\"
files_big = os.listdir(big_path)
files_big.remove('meta.json')

for big in files_big:
    print(big)
    Path_NewFolder = big_path + big + "\\output\\"
    if not os.path.exists(Path_NewFolder):
         os.makedirs(Path_NewFolder)
    # path to dataset
    my_path = "C:\\Users\\leoni\\Desktop\\Uni\\CSE_projekt\\fsoco_bounding_boxes_train\\" + big
    save_path = "C:\\Users\\leoni\\Desktop\\Uni\\CSE_projekt\\fsoco_bounding_boxes_train\\" + big + "\\output\\"
    ann_path = my_path + "\\ann"

    # gets all the files in a list
    files = [join(ann_path, f) for f in listdir(ann_path) if isfile(join(ann_path, f))]
    num = 0
    # loop through all files
    for i, file in enumerate(files):
        # open the annotationfile
        with open(file, "r", encoding="utf-8") as content:
            #print(content)
            parsed = json.load(content)
            #von json zu image file und im Ordner \img statt \\ann abspeichern
            img_path = file.replace("\\ann", "\img").replace(".json", "")
            #image einlesen
            img = cv2.imread(img_path)

            # loop through cones contained in file
            for cone in parsed["objects"]:
                #object in der json datei geöffnet
                cone_type = cone["classTitle"]
                points = cone["points"]["exterior"]

                left = points[0][0]
                right = points[1][0]
                low = points[0][1]
                high = points[1][1]

                width = right - left
                height = high - low

                #bild zuschneiden
                cropped_img = img[low:low+height, left:left+width]
                path = save_path + cone_type + str(num) + ".jpg"
                #Bild abspeichern
                cv2.imwrite(path, cropped_img)
                num = num + 1