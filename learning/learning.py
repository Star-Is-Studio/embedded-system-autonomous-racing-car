import tensorflow as tf
import numpy as np
import cv2
from copy import deepcopy

np.random.seed(20181209)

# 이미지와 label을 동일한 순서로 랜덤하게 섞어준다.
def shuffle(images, labels):
    per = np.random.permutation(len(images))

    permuted_images = []
    permuted_labels = []
    for i in per:
        permuted_images.append(images[i])
        permuted_labels.append(labels[i])
    
    return permuted_images, permuted_labels

#데이터에서 begin의 인덱스 부터 batch size 만큼 minibatch 를 가져온다
#minibatch 데이터와 반환한 마지막 데이터의 다음 인덱스 값 반환
def next_batch(data, begin, batch_size):
    size = len(data)
    minibatch = []
    minibatch.append(data[begin%size])
    for data_idx in range(begin + 1, begin+batch_size):
        minibatch.append(data[data_idx%size])
    return np.array(minibatch), (begin+batch_size)%size

IMAGE_WIDHT = 64
IMAGE_HEIGHT = 30

# 데이터의 입력값을 0~255 정수에서 -1.0~1.0 상의 실수로 바꾸고
# 1차원 벡터로 바꿔준다
def input_data_encoding(data):
    data = np.array(data)
    data = data.reshape(len(data), data[0].size)
    data = data / 255
    data = 2*(data - 0.5)
    return data

#데이터의 타겟을 one_hot으로 인코딩한다
def one_hot_encoding(target):
    target_oh = []
    tmp = [0,0,0]
    for i in range(len(target)):
        val = target[i]
        tmp[val] = 1
        target_oh.append(deepcopy(tmp))
        tmp[val] = 0
    return np.array(target_oh)

# 이미지와 label을 읽어드림
images = []
for i in range(155):
    image = cv2.imread("road/road%03d.jpg"%i, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMAGE_WIDHT, IMAGE_HEIGHT), interpolation =cv2.INTER_LINEAR)
    images.append(image)

with open("road/labels.txt", "r") as f:
    data = f.read()
    labels = data.split("\n")

for i in range(len(labels)):
    labels[i] = int(labels[i])

# 이미지 좌우반전을 통해 data augmentation
flipped_images = []
for image in images:
    flipped_images.append(cv2.flip(image, 1))

flipped_labels = []
for label in labels:
    if label == 0:
        flipped_labels.append(2)
    elif label == 1:
        flipped_labels.append(1)
    else:
        flipped_labels.append(0)

images.extend(flipped_images)
labels.extend(flipped_labels)

# 입력 이미지와 타겟 데이터를 위에 정의된 함수로 인코딩
images = input_data_encoding(images)
labels = one_hot_encoding(labels)

# 데이터의 순서를 임의로 섞음
images, labels = shuffle(images, labels)

# 신경망 정의
num_input = IMAGE_HEIGHT*IMAGE_WIDHT
x = tf.placeholder(tf.float32, [None, num_input])

num_units1 = 16
num_units2 = 16

# 첫 번째 히든레이어
w1 = tf.Variable(tf.truncated_normal((num_input, num_units1)))
b1 = tf.Variable(tf.constant(0.1, shape = [num_units1]))
hidden1 = tf.nn.relu(tf.matmul(x, w1) - b1)

keep_prob = tf.placeholder(tf.float32)
hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)

# 두 번쨰 히든 레이어
w2 = tf.Variable(tf.truncated_normal((num_units1, num_units2)))
b2 = tf.Variable(tf.constant(0.1, shape = [num_units2]))
hidden2 = tf.nn.relu(tf.matmul(hidden1_dropout, w2) - b2)
hidden2_dropout = tf.nn.dropout(hidden2, keep_prob)

# 소프트맥스 함수
w0 = tf.Variable(tf.zeros((num_units2, 3)))
b0 = tf.Variable(tf.zeros([3]))
p = tf.nn.softmax(tf.matmul(hidden2_dropout, w0) - b0)

# 오차함수, 트레이닝 알고리즘, 정답률 정의
t = tf.placeholder(tf.float32, [None, 3])
loss = -tf.reduce_sum(t*tf.log(p))
train_step = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.argmax(p,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 전체 데이터에 20%를 test data로 둠
split_point = int(len(images)*0.8)
train_x = images[:split_point]
train_t = labels[:split_point]
test_x = images[split_point:]
test_t = labels[split_point:]

# 세션 준비 후 파라미터 최적화
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

SAVE_DIR = "save/"

begin = 0
for i in range(1, 2001):
    batch_size = 10
    batch_xs, _ = next_batch(train_x, begin, batch_size)
    batch_ts, begin = next_batch(train_t, begin, batch_size)
    sess.run(train_step, feed_dict = {x:batch_xs, t: batch_ts, keep_prob: 0.5})
    if i%100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict = {x: test_x, t: test_t, keep_prob: 1.0})
        print("Step: %d, Loss: %f, Accuracy: %f"% (i, loss_val, acc_val))
        saver.save(sess, "%ssession"%SAVE_DIR, global_step = i)
        if acc_val > 0.7:
            break