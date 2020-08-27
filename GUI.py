import tkinter as tk
from PIL import Image, ImageTk
import cv2
import sys
import os
import pickle
import numpy as np
import time
import array
import operator

import pyaudio
import wave

import scipy.signal
from scipy.signal import argrelextrema
import tkinter.messagebox

from sklearn.cluster import KMeans
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_float

FORMAT_LEN = 0
isNotPaused = True

current_type = ''
current_frame = 0

frame_matrix_list = {}
selected_image_dict = {}

frame_total_num = 0
test_file_path = ''


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


def set_keyFrame(img_name='null'):
	global current_type, current_frame, isNotPaused
	# current_input = str(testInput.get())
	current_input = img_name

	if not isNotPaused:
		isNotPaused = True

	current_type = current_input[0:5]
	if current_type == 'image':
		current_frame = int(current_input[6:])
		if current_type not in frame_matrix_list:
			pass
		# matrix_list = frame_matrix_list[current_type]
		new_image = Image.fromarray(selected_image_dict[current_input], 'RGB')
		new_cover = ImageTk.PhotoImage(image=new_image)
		play_area.configure(image=new_cover)
		play_area.image = new_cover
	elif current_type == 'video':
		current_type = current_input[0:6]
		current_frame = int(current_input[7:])
		# if video not read in memory yet
		if current_type not in frame_matrix_list:
			# frame_matrix_list[current_type] = []
			cur_mtx = []
			fileList = os.listdir(test_file_path + current_type + '/')
			for i in range(len(fileList) - 2):
				name = 'image-' + str(i + 1).zfill(4) + '.rgb'
				cur_mtx.append(get_matrix(test_file_path + current_type + '/' + name))
			frame_matrix_list[current_type] = cur_mtx
		matrix_list = frame_matrix_list[current_type]
		newImage = Image.fromarray(matrix_list[current_frame - 1], 'RGB')
		newCover = ImageTk.PhotoImage(image=newImage)
		play_area.configure(image=newCover)
		play_area.image = newCover

	# frame_name = frame_name.zfill(FORMAT_LEN)

	# newImage = Image.fromarray(matrix_list[current_frame-1], 'RGB').resize((1080, 500))


# 点击播放按钮，从封面的该帧处开始播放
def play_video():
	global current_frame, isNotPaused

	if current_type == 'image':
		tk.messagebox.showinfo("can not play video", 'it is just an image')
		return

	if not isNotPaused:
		isNotPaused = True

	if current_frame == 0:
		tk.messagebox.showinfo("ooops", 'you should click the synopsis firstly:)')
		return
	else:
		print(current_type)

		tic = time.time()

		video_path = test_file_path + "/" + current_type
		audio_path = video_path + "/audio.wav"

		fileList = os.listdir(video_path)
		video_total_num = int(len(fileList) - 1)

		time_total = 0.0
		# open wave file
		wave_obj = wave.open(audio_path, 'rb')
		chunk = int(wave_obj.getnframes() / video_total_num) + 1

		# initialize audio
		py_audio = pyaudio.PyAudio()
		audio_stream = py_audio.open(format=py_audio.get_format_from_width(wave_obj.getsampwidth()),
									 channels=wave_obj.getnchannels(),
									 rate=wave_obj.getframerate(),
									 output=True)
		pos = int((wave_obj.getnframes() / video_total_num) * current_frame)
		# initial position of audio
		wave_obj.setpos(pos)

		video_matrix_list = frame_matrix_list[current_type]

		for i in range(current_frame, len(video_matrix_list)):
			if isNotPaused:
				t1 = time.time()
				# newImage = Image.fromarray(video_matrix_list[i-1], 'RGB').resize((1080, 500))
				newImage = Image.fromarray(video_matrix_list[i - 1], 'RGB')
				newCover = ImageTk.PhotoImage(image=newImage)
				play_area.configure(image=newCover)
				play_area.image = newCover
				play_area.update()
				current_frame = i

				t2 = time.time()
				time_total += (t2 - t1) * 1000

				data = wave_obj.readframes(chunk)
				audio_stream.write(data)

				# cv2.waitKey(4)
			else:
				break

		toc = time.time()
		print("total running time = " + str(toc - tic) + "  s")
		avg_time = time_total / video_total_num
		print("avg processing time = " + str(avg_time) + "  ms")

		audio_stream.close()
		py_audio.terminate()
		wave_obj.close()

		# my_video = cv2.VideoCapture('./img/video_4.avi')
		# # 【如果是测FPS的话，三个视频都是29.97，即CIF格式视频的最大FPS】
		# # global isNotPaused
		# rate = my_video.get(5)
		# FrameNumber = my_video.get(7)
		# duration = FrameNumber/rate
		# print(duration)

		# while my_video.isOpened:
		#     ret, frame = my_video.read()
		#     if ret == True and isNotPaused:
		#         newImage = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
		#         newImage = Image.fromarray(newImage).resize((1080, 500))
		#         newCover = ImageTk.PhotoImage(image=newImage)
		#         play_area.configure(image=newCover)
		#         play_area.image = newCover
		#         play_area.update()
		#     else:
		#         break


def pause_video():
	global isNotPaused
	if isNotPaused:
		isNotPaused = False
	else:
		isNotPaused = True
		play_video()


def stop_video():
	global current_type, current_frame, isNotPaused

	if not isNotPaused:
		isNotPaused = True
	else:
		isNotPaused = False

	current_type = ''
	current_frame = 0

	newImage = Image.open('./img/stop_image.jpg').resize((1080, 500))
	newCover = ImageTk.PhotoImage(image=newImage)
	play_area.configure(image=newCover)
	play_area.image = newCover


def img_on_click(event):
	print("current click position：", event.x, event.y)
	cur_width = event.x
	img_name = meta_dict['img_name_list'][cur_width // chunk_width]
	set_keyFrame(img_name=img_name)


def read_video_image():
	print('~~~~~~~~~~ read videos and images into memory ~~~~~~~~~~~~~~~')
	for folder_name in os.listdir(test_file_path):
		if folder_name[0:5] == 'video':
			cur_mtx = []
			file_list = os.listdir(test_file_path + folder_name + '/')
			for i in range(len(file_list) - 2):
				name = 'image-' + str(i + 1).zfill(4) + '.rgb'
				cur_mtx.append(get_matrix(test_file_path + folder_name + '/' + name))
			frame_matrix_list[folder_name] = cur_mtx

	for file_name in os.listdir(selected_image_path):
		if file_name[-4:] != '.jpg':
			continue
		selected_image_dict[file_name[0:-4]] = cv2.imread(selected_image_path + file_name)

	print('~~~~~~~~~ reading ends ~~~~~~~~~~~')


if __name__ == '__main__':
	test_file_path = 'StudentsUse_Dataset_Armenia/'
	selected_image_path = 'selected_image/'
	syn_img_path = 'try18pic_smooth.png'
	meta_file = open('synopsis_metadata', 'rb')
	meta_dict = pickle.load(meta_file)
	chunk_width = meta_dict['single_pic_width']

	print(meta_dict)
	read_video_image()


	Root = tk.Tk()
	Root.title("csci576_GUI_Player")
	Root.geometry("1260x750")
	Root['bg'] = '#ffffff'

	play_area = tk.Label(Root, text='Display Here', bg='yellow')
	play_area.place(x=0, y=0, width=1260, height=400)

	iconImage_play = Image.open('./img/play.ico').resize((64, 64))
	iconBtn_play = ImageTk.PhotoImage(image=iconImage_play)
	play_Btn = tk.Button(image=iconBtn_play, cursor='hand2', command=play_video)
	play_Btn.place(x=50, y=500)

	iconImage_pause = Image.open('./img/pause.ico').resize((64, 64))
	iconBtn_pause = ImageTk.PhotoImage(image=iconImage_pause)
	pause_Btn = tk.Button(image=iconBtn_pause, cursor='hand2', command=pause_video)
	pause_Btn.place(x=200, y=500)

	iconImage_stop = Image.open('./img/stop.ico').resize((64, 64))
	iconBtn_stop = ImageTk.PhotoImage(image=iconImage_stop)
	stop_Btn = tk.Button(image=iconBtn_stop, cursor='hand2', command=stop_video)
	stop_Btn.place(x=350, y=500)

	# testInput = tk.Entry(show=None)
	# testInput.place(x=750, y=520)
	# test_Btn = tk.Button(text="set key_Frame",
	#                     width=15,
	#                     height=2,
	#                     command=set_keyFrame)
	# test_Btn.place(x=850, y=520)

	# synopsis_area = tk.Label(Root, text='synopsis Here', bg='green')
	syn_image = Image.open(syn_img_path)
	newCover = ImageTk.PhotoImage(image=syn_image)
	synopsis_area = tk.Label(Root, image=newCover)
	synopsis_area.place(x=0, y=600, width=1260, height=150)
	synopsis_area.bind("<Button-1>", img_on_click)
	# syn_image = Image.fromarray(matrix_list[current_frame - 1], 'RGB').resize((1080, 500))
	# synopsis_area.configure(image=newCover)
	# synopsis_area.image = newCover

	Root.mainloop()
