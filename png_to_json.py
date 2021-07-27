import numpy as np
from PIL import Image
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
import io
import base64
import cv2 as cv
import json
import os
import multiprocessing
from functools import partial
from multiprocessing import Pool
import tqdm
#from imantics import Polygons, Mask

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory

    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]


    return listOfFiles

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)

    sub_mask=np.uint8(sub_mask*255)

    # cv.namedWindow("Display window", cv.WINDOW_NORMAL)
    # cv.imshow( "Display window",cv.UMat(sub_mask))
    # cv.waitKey(0)

    ret, thresh = cv.threshold( sub_mask, 127, 255, 0)
    contours, hierarchy =cv.findContours(thresh,cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE )
    #if len(contours)!=0:
    #    hierarchy=hierarchy.get()[0]
    segmentations=[]
    areas=[]
    bboxs=[]
    h=[]
    contour_empty=[]
    for i in range(len(contours)):
        index_hole = 0
        #contour_skit=contours[0]

        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        contour=contours[i]
        contour = np.vstack(contour).squeeze()
        if len(contour)>2:

            if hierarchy[0][i][3] < 0:

                # cimage = cv.cvtColor(sub_mask, cv.COLOR_GRAY2BGR)
                # cv.drawContours(cimage, [contour], 0, (0, 255, 0), 10)
                # cv.namedWindow("image", cv.WINDOW_NORMAL)
                # cv.imshow('image', cimage)
                # cv.waitKey(0)


                # for i in range(len(contour)):
                #     row, col = contour[i]
                #     contour[i] = (col - 1, row - 1)

                # Make a polygon and simplify it


                segmentation = contour
                # Combine the polygons to calculate the bounding box and area
                x, y, max_x, max_y = cv.boundingRect(contour)
                width = max_x - x
                height = max_y - y
                bbox = (x, y, width, height)
                area = cv.contourArea(contour)
                segmentations.append(segmentation)
                areas.append(area)
                bboxs.append(bbox)
                h.extend([i])


            else:
                cont=hierarchy[0][i][3]
                if not(cont in contour_empty): #test if hole is enpty contour


                    #cont = cont - len([i for i in contour_empty if i < cont])  # delete all enpty contour


                    ###try 1 to join contours with hole
                    if cont in h: #search in hole is in a old contour
                        for hi in range(len(h)): #search index of contour with a hole
                            if cont==h[hi]:
                                index_hole=hi
                        segmentation_hole = np.array(contour).tolist()

                        #end_point = segmentations[cont][-1:]

                        near_idx=closest_node(segmentation_hole[0],segmentations[index_hole]) #search the nearest point

                        segmentations[index_hole]=np.concatenate([segmentations[index_hole][:near_idx],segmentation_hole,segmentations[index_hole][near_idx:]])

                        area_hole = cv.contourArea(contour)
                        areas[index_hole] = areas[index_hole] - area_hole


                        # cimage = cv.cvtColor(sub_mask, cv.COLOR_GRAY2BGR)
                        # pts = [np.array(contour, dtype=np.int32)]
                        # # pts.reshape((-1, 1, 2))
                        # cv.drawContours(cimage, pts, 0, (0, 255, 0), 10)
                        # cv.namedWindow("image", cv.WINDOW_NORMAL)
                        # cv.imshow('image', cimage)
                        # cv.waitKey(0)
                        # cimage = cv.cvtColor(sub_mask, cv.COLOR_GRAY2BGR)
                        # pts = [np.array(segmentations[index_hole], dtype=np.int32)]
                        # # pts.reshape((-1, 1, 2))
                        # cv.drawContours(cimage, pts, 0, (0, 0, 255), 5)
                        # cv.namedWindow("image", cv.WINDOW_NORMAL)
                        # cv.imshow('image', cimage)
                        # cv.waitKey(0)
                        # print(index_hole)



        else:
            contour_empty.append(i)
    # cimage = cv.cvtColor(sub_mask, cv.COLOR_GRAY2BGR)
    # pts = [np.array(segmentations[0], dtype=np.int32)]
    # # pts.reshape((-1, 1, 2))
    # cv.drawContours(cimage, pts, 0, (0, 0, 255), 3)
    # cv.namedWindow("image", cv.WINDOW_NORMAL)
    # cv.imshow('image', cimage)
    # cv.waitKey(0)

    return segmentations,areas,bboxs

def image_to_json(folder_picture, folder_png,labels,listOfFiles, i):
    l_path = len(str(folder_picture))
    picture = listOfFiles[i][l_path + 1:-3]

    image_png = os.path.join(folder_png, picture + "png")
    image_jpg = os.path.join(folder_picture, picture + "jpg")

    image_label_open = Image.open(image_png).convert('RGB')
    with open(image_jpg, "rb") as imageFile:
        str_image = base64.b64encode(imageFile.read()).decode("utf-8")

    is_crowd = 0

    # These ids will be automatically increased as we go
    annotation_id = 1
    image_id = 1

    # Create the annotations
    annotations = []
    image_label=np.asarray(image_label_open)
    for category_id in range(len(labels)):

        #mask=np.where(image_label==np.transpose(labels[category_id]['color']),1,0)
        mask=(image_label == labels[category_id]['color']).all(-1)
        if  mask.any()!=False:
            segmentations,areas,bboxs = create_sub_mask_annotation(mask, image_id, category_id, annotation_id, is_crowd)

            for i in range(0,np.size(areas)):
                annotation = {
                    'label': labels[category_id]['name'],
                    'line_color': None,
                    'fill_color': None,
                    'points': segmentations[i].tolist(),
                    'iscrowd': is_crowd,
                    'image_id': image_id,
                    'id': annotation_id,
                    'bbox': bboxs[i],
                    'shape_type': "polygon",
                    'area': areas[i],
                    "flags": {}
                    }
                annotations.append(annotation)
            annotation_id += 1

    indent=1
    height=np.size(image_label,0)
    width=np.size(image_label,1)

    im = Image.fromarray(image_label)
    #final json
    json_data = {
        'version': '3.16.7',
        'flags': {},
        'shapes': annotations,
        'lineColor': [0,255,0,128],
        'fillColor': [255,0,0,128],
        'imagePath': image_jpg,
        'imageData': json.dumps(str_image,indent=indent)[1:-1],
        'imageHeight': height,
        'imageWidth': width
        }
    #'imageData': json.dumps(str_image,indent=indent),
    str_ = json.dumps(json_data, indent=indent)
    #save json
    with io.open(os.path.join(folder_picture,picture+"json"), 'w', encoding='utf8') as output:
        if len(str_) > 0:

            output.write(str_)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="labelme annotation to coco data json file."
    )
    parser.add_argument(
        "--config",
        help="config json file with labels names",
        type=str,
    )
    parser.add_argument(
        "--folder_picture", help="folder where to find jpeg image", type=str,
    )
    parser.add_argument(
        "--folder_png",
        help="Directory to label images.",
        type=str,
    )
    args = parser.parse_args()
    folder_config = args.config
    folder_picture = args.folder_picture
    folder_png = args.folder_png

    # folder_config = "C:/Users/Guillaume/Documents/Python/AutonomousCar/Data/Mapilary/config.json "
    # folder_picture = "C:/Users/Guillaume/Documents/Python/AutonomousCar/Data/Mapilary/images/train"
    # folder_png = "C:/Users/Guillaume/Documents/Python/AutonomousCar/Data/Mapilary/labels/train"

    print("PNG: "+str(folder_png))
    print("picture: " + str(folder_picture))
    with open(folder_config) as config_file:
        config = json.load(config_file)
    # in this example we are only interested in the labels
    labels = config['labels']
    listOfFiles = getListOfFiles(folder_picture)
    listOfFiles=[file for file in listOfFiles if file[-3:]=="jpg"]
    l=len(listOfFiles)


    image_to_json_partial=partial(image_to_json,folder_picture,folder_png,labels,listOfFiles)



    # for i in range(0,l):
    #     p = multiprocessing.Process(target=image_to_json_partial, args=(listOfFiles[i],))
    #     print("\r \r{0}".format(str(i) + " / " + str(l)), end='')
    #     #a=prepImputML_partial(i)
    #     p.start()
    #    #p.join() #pas obliagtoire
    # p.terminate()

    # for j in range(l):
    #     image_to_json_partial(j)

    pool = Pool()
    for _ in tqdm.tqdm(pool.imap_unordered( image_to_json_partial, range(l)), total=l):
        pass
