from picamera import PiCamera

IMAGE_NAME = "./tmp_image.jpg"

def capture():
	camera = PiCamera()
	
	# set resolution
	camera.resolution = (1280, 600) 
	
	# capture
	camera.start_preview(fullscreen=False, window = (40,20,640,480))
	
	#camera.capture(IMAGE_NAME, use_video_port = True)

	#camera.close
	while True:
		try:
			pass
		except:
			break

capture()

