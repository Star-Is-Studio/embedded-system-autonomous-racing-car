"""
임베디드 시스템 설계 과제
자율 주행 자동차
2014920002	김세연
2014920009	김현우
"""
# -*- coding: utf-8 -*-
import numpy as np
import cv2

import sys
import os
import time
import math
import getopt

from picamera import PiCamera
from picamera.array import PiRGBArray
import RPi.GPIO as gpio

from Image import *
from Utils import *

N_SLICES = 5 # 이미지 분할 수
IMAGE_WIDTH = 1280 # 카메라에서 읽어들이는 이미지 넓이
IMAGE_HEIGHT = 600 # 카메라에서 읽어들이는 이미지 높이
CROP_HEIGHT = int(IMAGE_HEIGHT/N_SLICES) # 분할된 이미지의 높이

IMAGE_LOG_DIR = "images_log/"
TMP_IMAGE_NAME = "./tmp_image.jpg"
DO_SAVE_LOG = False

MAX_BASE_SPEED = 0.1 # 기준 바퀴 속도의 최댓 값(단위 : 초)
MAX_DIFF_SPEED = 0.4 # 두 바퀴 사이의 속도의 최댓 값(단위 : 초)
WEIGHT_ORIGIN = 0.6 # 중심으로부터 떨어진 도로의 거리의 보정 비중

DEBUG_MODE = False

try:
	opts, args = getopt.getopt(sys.argv[1:], "d", ["base=", "diff="])
except getopt.GetoptError as err:
	print(err)

for opt,arg in opts:
	if opt == "-d":
		DEBUG_MODE = True
	elif opt == "--base":
		MAX_BASE_SPEED = float(arg)
	elif opt == "--diff":
		MAX_DIFF_SPEED = float(arg)

MOTOR_A1 = 27 #forward
MOTOR_A2 = 17 #back
MOTOR_B1 = 23 #forward
MOTOR_B2 = 24 #back

gpio.setmode(gpio.BCM)
gpio.setup(MOTOR_A1, gpio.OUT)
gpio.setup(MOTOR_A2, gpio.OUT)
gpio.setup(MOTOR_B1, gpio.OUT)
gpio.setup(MOTOR_B2, gpio.OUT)

gpio.output(MOTOR_A1, False)
gpio.output(MOTOR_A2, False)
gpio.output(MOTOR_B1, False)
gpio.output(MOTOR_B2, False)

def motorCtrl(left, right):
    """
    rotate left motor for left(s), and right motor for right(s)
    :param left: 왼쪽 모터를 회전시키는 시간(초)
    :param right: 오른쪽 모터를 회전시키는 시간(초)
    """
   
    t = abs(left-right)
    if left>right:
    	left = left - t
    	right = 0
    elif right>left:
    	right = right-t
    	left=0
    gpio.output(MOTOR_A1, True)
    gpio.output(MOTOR_A2, False)
    gpio.output(MOTOR_B1, True)
    gpio.output(MOTOR_B2, False)
    time.sleep(t)
    
    # MOTOR_A left 초간 회전
    
    gpio.output(MOTOR_A1, True)
    gpio.output(MOTOR_A2, False)
    time.sleep(left)
    gpio.output(MOTOR_A1, False)
    gpio.output(MOTOR_A2, False)

    # MOTOR_B left 초간 회전
    gpio.output(MOTOR_B1, True)
    gpio.output(MOTOR_B2, False)
    time.sleep(right)
    gpio.output(MOTOR_B1, False)
    gpio.output(MOTOR_B2, False)
	
	
def right(t):
	gpio.output(MOTOR_B1, True)
	gpio.output(MOTOR_B2, False)
	gpio.output(MOTOR_A1, False)
	gpio.output(MOTOR_A2, True)
	time.sleep(t)
	gpio.output(MOTOR_B1, False)	
	gpio.output(MOTOR_B2, False)
	gpio.output(MOTOR_A1, False)
	gpio.output(MOTOR_A2, False)
	

def left(t):
	gpio.output(MOTOR_B1, False)
	gpio.output(MOTOR_B2, True)
	gpio.output(MOTOR_A1, True)
	gpio.output(MOTOR_A2, False)
	time.sleep(t)
	gpio.output(MOTOR_B1, False)	
	gpio.output(MOTOR_B2, False)
	gpio.output(MOTOR_A1, False)
	gpio.output(MOTOR_A2, False)

def straight(t):
	gpio.output(MOTOR_A1, True)
	gpio.output(MOTOR_A2, False)
	gpio.output(MOTOR_B1, True)
	gpio.output(MOTOR_B2, False)
	time.sleep(t)
	gpio.output(MOTOR_A1, False)
	gpio.output(MOTOR_A2, False)
	gpio.output(MOTOR_B1, False)
	gpio.output(MOTOR_B2, False)
	
def backward():
	gpio.output(MOTOR_A1, False)
	gpio.output(MOTOR_A2, True)
	gpio.output(MOTOR_B1, False)
	gpio.output(MOTOR_B2, True)
	time.sleep(0.2)
	gpio.output(MOTOR_A1, False)
	gpio.output(MOTOR_A2, False)
	gpio.output(MOTOR_B1, False)
	gpio.output(MOTOR_B2, False)

def main():
	camera = PiCamera()
	camera.resolution = (IMAGE_WIDTH, IMAGE_HEIGHT)
	rawCapture = PiRGBArray(camera, size = (IMAGE_WIDTH, IMAGE_HEIGHT))
	
	time.sleep(0.1)

	while True:
		try:
			#N_SLICES만큼 이미지를 조각내서 Images[] 배열에 담는다
			Images=[]
			for _ in range(N_SLICES):
				Images.append(Image())

			camera.capture(rawCapture, format = "bgr", use_video_port=True)
			frame = rawCapture.array
			
		    # 이미지를 조각내서 윤곽선을 표시하게 무게중심 점을 얻는다
			points = SlicePart(frame, Images, N_SLICES)		
			
			topPoint = points[2][0]		
			botPoint = points[-1][0]
			dx = topPoint - botPoint
			dy = 2*CROP_HEIGHT			
			
			#print("dx: %d, dy: %d"%(dx,dy))
			roadAngle = math.atan2(dy, dx)
			dAngle = roadAngle - math.pi/2
			max_dangle = math.atan(IMAGE_WIDTH/CROP_HEIGHT/2)
			curveRatio = dAngle/max_dangle # 도로가 휜 정도, 양수면 좌로 휨, 음수면 우로 huim
			#print("roadAngle: %f" % roadAngle)
			#print("dAngle: %f, max_dangle:%f, curveRation: %f"%(dAngle, max_dangle, curveRatio))

			baseSpeed = MAX_BASE_SPEED * (1 - abs(curveRatio))# 기준 속도 계산, 도로가 많이 휘었을 수록 작아짐
			diffRatio = Images[-1].diff/IMAGE_WIDTH/2 # 도로의 가장 밑의 부분을 기준으로 중심에서 떨어진 비율, 양수면 좌측에, 음수면 우측에
			diffSpeed = MAX_DIFF_SPEED*((1-WEIGHT_ORIGIN)*curveRatio + WEIGHT_ORIGIN*diffRatio) # curveRatio와 diffRatio를 비중에 따라 합쳐 두 모터 사이의 속도 차이를 구함

			#print("base: %f"%baseSpeed)
			#print("diff: %f"%diffSpeed)
			
			if diffSpeed > 0: # 좌회전
				left(diffSpeed)
				straight(baseSpeed)
			else: # 우회전
				right(-diffSpeed)
				straight(baseSpeed)
				
			#print("base: %f, diff: %f"%(baseSpeed, diffSpeed))
	        	# 구한 모터 스피드만큼 모터를 회전시킴
			#motorCtrl(leftSpeed, rightSpeed)
			
			if DEBUG_MODE:
				# 조각난 이미지를 한개로 다시 합침
				frame = RepackImages(Images)
				# 도로를 근사한 직선 그림
				frame = cv2.line(frame, (topPoint,points[2][1] + 2*CROP_HEIGHT), (points[-1][0],points[-1][1]+4*CROP_HEIGHT), (0,0,255), 3)
				
				# 모터 제어 정보 이미지에 표시
				frame = cv2.putText(frame, "base speed: %f, diff speed: %f"%(baseSpeed, diffSpeed), (10, IMAGE_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
				cv2.imshow("Camera image", frame)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
			
			
			rawCapture.truncate(0)
			if DO_SAVE_LOG:
				# 이미지를 IMAGE_LOG_DIR에 저장
				# TODO
				pass
			else:
				pass			    
		except KeyboardInterrupt:	
			print("Key Interrput")		
			camera.close()
			cv2.destroyAllWindows()
			gpio.cleanup()
			break
		except BaseException as e:
			print(e)
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
					
			camera.close()
			cv2.destroyAllWindows()
			gpio.cleanup()
			break
		
if __name__ == "__main__":
	main()
