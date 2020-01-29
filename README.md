# labels
Labels image/picture for deeplearning 


# labels_to_png.py

Transforms all png pictures used to labels images to json files. These json files work perfectly with label me. You can use after labelme2coco.py to do a COCO file.

Exemple to use :  python png_to_json_4.py --folder_picture D:\images --folder_png D:\label --config D:\python\AutonomousCar\Data\cityscapes\config.json 

you can find an exemple of config.json file in the repetory. It is defined color used and name of label.
