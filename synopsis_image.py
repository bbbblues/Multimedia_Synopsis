
import cv2
import numpy as np
import time
from seam_carving import *
import math
import pickle
import os
import face_recognition
from functools import cmp_to_key


def compare_filename(n1, n2):
    if int(n1[5]) == int (n2[5]):
        return int(n1[7:-4]) - int(n2[7:-4])
    else:
        return int(n1[5]) - int(n2[5])


def load_img(key_frame_path, img_path):
    img_array = []
    img_name_array = []
    frame_names = os.listdir(key_frame_path)
    if '.DS_Store' in frame_names:
        frame_names.remove('.DS_Store')
    frame_names = sorted(frame_names, key=cmp_to_key(compare_filename))
    # print(frame_names)
    for img_name in frame_names:
        if img_name[-4:] != '.jpg':
            continue
        img_array.append(cv2.imread(key_frame_path + img_name))
        img_name_array.append(img_name[:-4])

    for img_name in os.listdir(img_path):
        if img_name[-4:] != '.jpg':
            continue
        img_array.append(cv2.imread(img_path + img_name))
        img_name_array.append(img_name[:-4])

    # img1 = cv2.imread("./inputs/two_faces.jpg")
    # img_array.append(img1)
    # img2 = cv2.imread("./inputs/image-0091.jpg")
    # img_array.append(img2)
    #
    # img3 = cv2.imread("./inputs/image-0058.jpg")
    # img_array.append(img3)
    # img4 = cv2.imread("./inputs/image-0326.jpg")
    # img_array.append(img4)
    return img_array, img_name_array


def make_continuous(splitted_img):
    continuous = np.copy(splitted_img)
    for edge_ind in range(1, input_num):
        edge = edge_ind * new_width
        for delta in range(-smooth_radius, smooth_radius + 1):
            j = edge + delta
            smooth_kernel_r = math.ceil(max_smooth_kernel_r * (smooth_radius - abs(delta)) / smooth_radius)
            for i in range(new_height):
                smooth(i, j, continuous, splitted_img, smooth_kernel_r)

    return continuous


def smooth(i, j, continous, splitted_img, smooth_kernel_r):
    cnt = 0
    h = splitted_img.shape[0]
    w = splitted_img.shape[1]
    num_channel = splitted_img.shape[2]
    sum = np.zeros(num_channel)
    for di in range(-smooth_kernel_r, smooth_kernel_r + 1):
        for dj in range(-smooth_kernel_r, smooth_kernel_r + 1):
            newi = i + di
            newj = j + dj
            if newi < 0 or newi >= h:
                continue
            for c in range(num_channel):
                sum[c] += splitted_img[newi, newj, c]
            cnt += 1
    continous[i, j, :] = sum / cnt


def metadata_generation():
    meta_dict = {}
    meta_dict['single_pic_height'] = new_height
    meta_dict['single_pic_width'] = new_width
    # range_dict = {}
    # for i in range(input_num):
    #     range_dict[(i*new_width, new_width)] = 'xxx.rgb'
    # meta_dict['range_dict'] = range_dict
    meta_dict['img_name_list'] = img_names
    meta_file = open('synopsis_metadata', 'ab')
    pickle.dump(meta_dict, meta_file)
    meta_file.close()

if __name__ == '__main__':

    new_height = 150
    new_width = 70
    smooth_radius = 7
    max_smooth_kernel_r = 20  # average on 3*3 block

    imgs, img_names = load_img('key_frames/', 'selected_image/')
    input_num = len(imgs)
    out_array = []
    res_name = 'try18pic'

    print(input_num)

    start = time.time()
    for i, img in enumerate(imgs):
        # print(type(img))
        face_locations = face_recognition.face_locations(img)
        if len(face_locations) == 0:
            obj = SeamCarver(img, new_height, new_width)
            out_array.append(obj.get_out_image())
            print("Finish seam carving: " + str(i))
        else:
            # faces_array = []
            # for face_location in face_locations:
            top, right, bottom, left = face_locations[0]
            faceh = bottom - top
            facew = right - left
            face_image = img[top - faceh//2 : bottom + faceh//2, left - facew//2 : right + facew//2]
            # faces_array.append(face_image)
            # faces = np.stack(faces_array, axis=0)
            # concated = np.copy(faces[0])
            # for i in range(1, faces.shape[0]):
            #     concated = np.concatenate((concated, faces[i]), axis=0)
            # out_array.append(cv2.resize(concated, (new_width, new_height)))
            out_array.append(cv2.resize(face_image, (new_width, new_height)))
            print("Finish face resizing: " + str(i))

    out_imgs = np.stack(out_array, axis=0)
    concated = np.copy(out_imgs[0])
    for i in range(1, out_imgs.shape[0]):
        concated = np.concatenate((concated, out_imgs[i]), axis=1)

    cv2.imwrite(res_name + "_orig.png", concated.astype(np.uint8))
    res = make_continuous(concated)

    end = time.time()
    print("process time: " + str(end - start))

    cv2.imshow("res", res.astype(np.uint8))
    cv2.waitKey(0)
    cv2.imwrite(res_name + "_smooth.png", res.astype(np.uint8))

    metadata_generation()
