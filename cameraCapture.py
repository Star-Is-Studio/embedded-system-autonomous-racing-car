from picamera import PiCamera
from time import sleep



i = 0

while True:
	try:
		camera = PiCamera()
		camera.resolution = (1280,600)

		camera.start_preview(fullscreen = False,window = (100,20,640,480))
		while True:
			input()
			camera.capture('road%03d.jpg'%i,use_video_port=True)
			i +=1
		camera.stop_preview()
		camera.close()
	except:
		break


