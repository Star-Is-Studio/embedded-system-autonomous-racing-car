"""
임베디드 시스템 설계 과제
자율 주행 자동차
2014920002	김세연
2014920009	김현우
"""
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from copy import deepcopy

class Image:
    
	def __init__(self):
		self.image = None
		self.contourCenterX = 0
		self.MainContour = None
		self.diff = 0

	def Process(self):
		"""
		if self.image == None:
			return [0,0]
		else:
		"""
		#이미지를 흑백으로 변환한 뒤 Threshold 값을 기준으로 0 또는 1로 값을 정한다
		imgray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY) #Convert to Gray Scale
		ret, thresh = cv2.threshold(imgray,75,255,cv2.THRESH_BINARY_INV) #Get Threshold

		self.contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1] #Get contour

		if len(self.contours) == 0:
			return [0, 0]
		
		self.MainContour = max(self.contours, key=cv2.contourArea)

		self.height, self.width  = self.image.shape[:2]

		self.middleX = int(self.width/2) #Get X coordenate of the middle point
		self.middleY = int(self.height/2) #Get Y coordenate of the middle point

		if self.getContourCenter(self.MainContour) != 0:
			self.contourCenterX = self.getContourCenter(self.MainContour)[0]
		else:
			self.contourCenterX = 0

		self.diff =  self.middleX-self.contourCenterX

		#윤곽선은 초록색, 무게중심은 흰색 원, 그림의 중앙 지점은 빨간 원으로 표시
		cv2.drawContours(self.image,self.MainContour,-1,(0,255,0),3) #Draw Contour GREEN
		cv2.circle(self.image, (self.contourCenterX, self.middleY), 7, (255,255,255), -1) #Draw dX circle WHITE
		cv2.circle(self.image, (self.middleX, self.middleY), 3, (0,0,255), -1) #Draw middle circle RED

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(self.image,str(self.diff),(self.contourCenterX+20, self.middleY), font, 1,(200,0,200))
		cv2.putText(self.image,"Weight:%.3f"%self.getContourExtent(self.MainContour),(self.contourCenterX+20, self.middleY+35), font, 0.5,(200,0,200))
		return [deepcopy(self.contourCenterX), deepcopy(self.middleY)]

	def getContourCenter(self, contour):
		M = cv2.moments(contour)

		if M["m00"] == 0:
			return 0

		x = int(M["m10"]/M["m00"])
		y = int(M["m01"]/M["m00"])

		return [x,y]

	def getContourExtent(self, contour):
		area = cv2.contourArea(contour)
		x,y,w,h = cv2.boundingRect(contour)
		rect_area = w*h
		if rect_area > 0:
			return (float(area)/rect_area)  
                            
