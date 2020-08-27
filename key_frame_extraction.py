import scipy.signal
from scipy.signal import argrelextrema
from PIL import Image, ImageTk
import cv2
import sys
import os
import numpy as np
import array
import operator

from sklearn.cluster import KMeans
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_float

min_brightness_value = 10.0
max_brightness_value = 90.0
min_entropy_value = 1.0
max_entropy_value = 10.0


class Frame:
	"""class to hold information about each frame
	"""

	def __init__(self, id, diff):
		self.id = id
		self.diff = diff

	def __lt__(self, other):
		if self.id == other.id:
			return self.id < other.id
		return self.id < other.id

	def __gt__(self, other):
		return other.__lt__(self)

	def __eq__(self, other):
		return self.id == other.id and self.id == other.id

	def __ne__(self, other):
		return not self.__eq__(other)


class KeyFrame:
	"""class to hold information about each frame
	"""
	def __init__(self, matrix, path):
		self.path = path
		self.matrix = matrix


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


def get_format_len(video_path):
	global FORMAT_LEN, frame_total_num
	file_list = os.listdir(video_path)
	str_test = file_list[1]
	str_test = str_test[6:-4]

	# frame_total_num = int(len(file_list)-1)
	frame_total_num = int(len(file_list) - 2)  # for mac os
	FORMAT_LEN = len(str_test)


def getKeyFrame(file_name, file_path):
	# global frame_matrix_list

	# outPutDirName = './img/tempOutput'

	if not os.path.exists(outPutDirName):
		os.makedirs(outPutDirName)

	print("target video :" + file_name)
	print("frame save directory: " + outPutDirName)

	curr_frame = None
	prev_frame = None
	frame_diffs = []
	frames = []

	video_path = str(file_path + "/" + file_name)

	get_format_len(video_path)

	matrix_list = []

	for i in range(frame_total_num):

		frame_num = str(i + 1).zfill(FORMAT_LEN)
		frame_name = video_path + "/image-" + frame_num + '.rgb'

		frame_matrix = get_matrix(frame_name)

		matrix_list.append(frame_matrix)

		curr_frame = frame_matrix

		if curr_frame is not None and prev_frame is not None:
			diff = cv2.absdiff(curr_frame, prev_frame)
			diff_sum = np.sum(diff)
			diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
			frame_diffs.append(diff_sum_mean)

			frame = Frame(i, diff_sum_mean)
			frames.append(frame)
		prev_frame = curr_frame

	keyframe_id_set = set()

	# frame_matrix_list[str(file_name)] = matrix_list

	diff_array = np.array(frame_diffs)
	sm_diff_array = scipy.signal.savgol_filter(diff_array, 201, 5)

	frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
	for i in frame_indexes:
		keyframe_id_set.add(frames[i - 1].id)

	# use k-means to remove duplicates
	non_process_KFlist = []

	for idx in range(frame_total_num):
		if idx in keyframe_id_set:
			name = file_name + "-" + str(idx + 1) + ".jpg"
			keyframe_id_set.remove(idx)

			key_frame = KeyFrame(matrix_list[idx], name)
			non_process_KFlist.append(key_frame)

	temp_list = filter_step1(non_process_KFlist)
	temp_list = prepare_cluster_sets(temp_list)
	filtered_items = select_keyFrame(non_process_KFlist, temp_list)
	filtered_items.sort()

	for index in range(len(filtered_items)):
		image = Image.fromarray(non_process_KFlist[filtered_items[index]].matrix, 'RGB')
		image.save(outPutDirName + "/" + non_process_KFlist[filtered_items[index]].path)


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


def filter_step1(input_img_files):
	n_files = len(input_img_files)
	list_after = []
	for i in range(n_files):

		matrix = input_img_files[i].matrix

		brightness_score = get_brighness_score(matrix)
		entropy_score = get_entropy_score(matrix)

		if brightness_score > min_brightness_value and brightness_score < max_brightness_value and entropy_score > min_entropy_value and entropy_score < max_entropy_value:
			list_after.append(input_img_files[i])

	return list_after


# K-means clustering of frames
def prepare_cluster_sets(files):
	all_hists = []
	for img_file in files:
		img = cv2.cvtColor(img_file.matrix, cv2.COLOR_BGR2GRAY)
		hist = cv2.calcHist([img], [0], None, [256], [0, 256])
		hist = hist.reshape((256))
		all_hists.append(hist)

	# Kmeans clustering on the histograms
	kmeans = KMeans(n_clusters=3, random_state=0).fit(all_hists)
	labels = kmeans.labels_

	files_clusters_index_array = []
	for i in np.arange(3):
		index_array = np.where(labels == i)
		files_clusters_index_array.append(index_array)

	files_clusters_index_array = np.array(files_clusters_index_array)
	return files_clusters_index_array


def get_laplacian_scores(files, n_images):
	variance_laplacians = []
	for image_i in n_images:
		img_file = files[n_images[image_i]]
		img = cv2.cvtColor(img_file.matrix, cv2.COLOR_BGR2GRAY)

		# Calculating the blurryness of image
		variance_laplacian = cv2.Laplacian(img, cv2.CV_64F).var()
		variance_laplacians.append(variance_laplacian)

	return variance_laplacians


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
	test_file_path = './StudentsUse_Dataset_Armenia/'
	outPutDirName = './key_frames/'
	for file_name in os.listdir(test_file_path):
		if file_name[0:5] == 'video':
			getKeyFrame(file_name, test_file_path)