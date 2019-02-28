import RPi.GPIO as gpio
import time

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
	
def right():
	gpio.output(MOTOR_B1, True)
	gpio.output(MOTOR_B2, False)
	gpio.output(MOTOR_A1, False)
	gpio.output(MOTOR_A2, True)
	time.sleep(0.4)
	gpio.output(MOTOR_B1, False)	
	gpio.output(MOTOR_B2, False)
	gpio.output(MOTOR_A1, False)
	gpio.output(MOTOR_A2, False)
	

def left():
	gpio.output(MOTOR_B1, False)
	gpio.output(MOTOR_B2, True)
	gpio.output(MOTOR_A1, True)
	gpio.output(MOTOR_A2, False)
	time.sleep(0.4)
	gpio.output(MOTOR_B1, False)	
	gpio.output(MOTOR_B2, False)
	gpio.output(MOTOR_A1, False)
	gpio.output(MOTOR_A2, False)

def straight():
	gpio.output(MOTOR_A1, True)
	gpio.output(MOTOR_B1, True)
	time.sleep(0.4)
	gpio.output(MOTOR_A1, False)
	gpio.output(MOTOR_B1, False)



while True:
	try:
		right()
	except:
		gpio.cleanup()
		break
