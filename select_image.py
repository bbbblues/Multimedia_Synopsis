import cv2
import numpy as np
import time
import face_recognition
import os
import array

import scipy.signal
from scipy.signal import argrelextrema
from PIL import Image, ImageTk
import sys
import operator

max_img_num = 5

from sklearn.cluster import KMeans
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_float

min_brightness_value = 10.0
max_brightness_value = 90.0
min_entropy_value = 1.0
max_entropy_value = 10.0


def get_matrix(file_path):
	f = open(file_path, 'rb')
	# using 'B' to treat every number as byte
	a = array.array('B')
	a.fromfile(f, os.path.getsize(file_path) // a.itemsize)
	f.close()

	res = np.asarray(a)
	temp = np.hsplit(res, 3)
	redTemp = temp[0]
	greenTemp = temp[1]
	blueTemp = temp[2]

	res = np.column_stack((redTemp, greenTemp, blueTemp))
	res = res.reshape((288, 352, 3))
	return res


class SImage:
	def __init__(self, matrix, path):
		self.path = path
		self.matrix = matrix


def get_laplacian_scores(files, n_images):
	variance_laplacians = []
	for image_i in n_images:
		img_file = files[n_images[image_i]]
		img = cv2.cvtColor(img_file.matrix, cv2.COLOR_BGR2GRAY)

		# Calculating the blurryness of image
		variance_laplacian = cv2.Laplacian(img, cv2.CV_64F).var()
		variance_laplacians.append(variance_laplacian)

	return variance_laplacians


def get_brighness_score(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	_, _, v = cv2.split(hsv)
	sum = np.sum(v, dtype=np.float32)
	num_of_pixels = v.shape[0] * v.shape[1]
	brightness_score = (sum * 100.0) / (num_of_pixels * 255.0)
	return brightness_score


def get_entropy_score(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	entr_img = entropy(gray, disk(5))
	all_sum = np.sum(entr_img)
	num_of_pixels = entr_img.shape[0] * entr_img.shape[1]
	entropy_score = (all_sum) / (num_of_pixels)

	return entropy_score


# first step filter using brightness and entropy
def filter_step1(input_img_files):
	n_files = len(input_img_files)
	list_after = []
	for i in range(n_files):

		matrix = input_img_files[i].matrix

		brightness_score = get_brighness_score(matrix)
		entropy_score = get_entropy_score(matrix)

		if brightness_score > min_brightness_value and brightness_score < max_brightness_value \
				and entropy_score > min_entropy_value and entropy_score < max_entropy_value:
			list_after.append(input_img_files[i])

	return list_after


# K-means clustering of frames
def prepare_cluster_sets(files, class_num):
	all_hists = []
	# 计算每个矩阵图片的 histograms
	for img_file in files:
		img = cv2.cvtColor(img_file.matrix, cv2.COLOR_BGR2GRAY)
		hist = cv2.calcHist([img], [0], None, [256], [0, 256])
		hist = hist.reshape((256))
		all_hists.append(hist)

	# Kmeans clustering on the histograms
	kmeans = KMeans(n_clusters=class_num, random_state=0).fit(all_hists)
	labels = kmeans.labels_

	# categorize input labels
	files_clusters_index_array = []
	for i in np.arange(class_num):
		index_array = np.where(labels == i)
		files_clusters_index_array.append(index_array)

	files_clusters_index_array = np.array(files_clusters_index_array)
	return files_clusters_index_array

# get the index with lowest blurryness
def select_keyFrame(files, files_clusters_index_array):
	filtered_items = []

	clusters = np.arange(len(files_clusters_index_array))
	for cluster_i in clusters:
		curr_row = files_clusters_index_array[cluster_i][0]
		n_images = np.arange(len(curr_row))
		variance_laplacians = get_laplacian_scores(files, n_images)

		selected_frame_of_current_cluster = curr_row[np.argmax(variance_laplacians)]
		filtered_items.append(selected_frame_of_current_cluster)

	return filtered_items


if __name__ == '__main__':
	path = 'StudentsUse_Dataset_Armenia/image/'
	out_path = 'selected_image/'
	if not os.path.exists(out_path):
		os.makedirs(out_path)

	non_process_KFlist = []
	non_process_KFlist_no_face = []
	for file_name in os.listdir(path):
		if file_name[-4:] != '.rgb':
			continue
		cur_img = get_matrix(path + file_name)
		face_locations = face_recognition.face_locations(cur_img)
		if len(face_locations) == 0:
			cur_name = file_name[:-4] + '.jpg'
			non_process_KFlist_no_face.append(SImage(cur_img, cur_name))
		else:
			# cv2.imwrite(out_path + file_name[:-4] + '.jpg', cur_img)
			cur_name = file_name[:-4] + '.jpg'
			non_process_KFlist.append(SImage(cur_img, cur_name))

	temp_list = filter_step1(non_process_KFlist)
	temp_list = prepare_cluster_sets(temp_list, class_num=3)
	filtered_items = select_keyFrame(non_process_KFlist, temp_list)
	filtered_items.sort()

	for index in range(len(filtered_items)):
		image = Image.fromarray(non_process_KFlist[filtered_items[index]].matrix, 'RGB')
		image.save(out_path + "/" + non_process_KFlist[filtered_items[index]].path)

	temp_list_nf = filter_step1(non_process_KFlist_no_face)
	temp_list_nf = prepare_cluster_sets(temp_list_nf, class_num=2)
	filtered_items_nf = select_keyFrame(non_process_KFlist_no_face, temp_list_nf)
	filtered_items_nf.sort()

	for index in range(len(filtered_items_nf)):
		image = Image.fromarray(non_process_KFlist_no_face[filtered_items_nf[index]].matrix, 'RGB')
		image.save(out_path + "/" + non_process_KFlist_no_face[filtered_items[index]].path)

