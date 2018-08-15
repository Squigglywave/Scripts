import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
from tf_trt_models.detection import download_detection_model, build_detection_graph
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import random

FILENAME = './coco.names'

DATA_DIR = './data/'
IMAGE_PATH = './data/waving_hands_004.jpg'

frozen_graph, input_names, output_names = build_detection_graph(
    config='/home/autoware/tensorrt_ws/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config',
    checkpoint='/home/autoware/tensorrt_ws/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt'
)

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

tf_sess = tf.Session(config=tf_config)

tf.import_graph_def(trt_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('classes:0')

print "\n**************************************************"
print "***************LOADED TENSORRT********************"
print "**************************************************\n"

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def create_dictionary(filename):
    generic_dict = {}
    r = lambda: random.randint(0, 255)
    with open(filename) as f:
    	for i, l in enumerate(f):
    		generic_dict[i] = (l[:-1], (r(), r(), r()))
    return generic_dict

def colors(n):
	color_list = []
	for i in range(n):
		r = lambda: random.randint(0, 255)
		color_list.append((r(), r(), r()))
	return color_list

print str(file_len(FILENAME))
box_text_colors = colors(file_len(FILENAME))

generic_dict = create_dictionary(FILENAME)

def detections_pub(image_data):
    fp = open(FILENAME, 'r')
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(image_data, "bgr8")
    height, width = img.shape[:2]
    image_resized = cv2.resize(img,(300,300))
    
    scores, boxes, classes = tf_sess.run([tf_scores, tf_boxes, tf_classes], feed_dict={
        tf_input: image_resized[None, ...]
    })
    
    boxes = boxes[0] # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    score_thresh = 0.60

    font = cv2.FONT_HERSHEY_SIMPLEX 
    output_text = []
    output_color = []
    output_score = []
    
    # plot boxes exceeding score threshold
    for i in range(len(scores)):
        if scores[i] > score_thresh:
            #image, text, lower left corner, font, font size, text color, thickness, type of line used??
            cv2.putText(img, generic_dict[int(classes[i])][0] + "  " + str(scores[i]), (int(boxes[i][1]*width)+10, int(boxes[i][0]*height)+20), font, 0.8, generic_dict[int(classes[i])][1], 2, cv2.LINE_AA)
            #Top left corner, bottom right corner, color, thickness
            cv2.rectangle(img,(int(boxes[i][1]*width), int(boxes[i][0]*height)), (int(boxes[i][3]*width), int(boxes[i][2]*height)), generic_dict[int(classes[i])][1], 3) 
           
    pub = rospy.Publisher('/detections/image_raw', Image, queue_size=1)
    rate = rospy.Rate(10)
    image_msg = Image()
    image_msg = bridge.cv2_to_imgmsg(img, "bgr8")
    pub.publish(image_msg)
    fp.close()

def detections_sub():
    rospy.init_node('listener', anonymous=True)
   # rospy.Subscriber("/nvcamera/camera/image_raw", Image, detections_pub)
    rospy.Subscriber("/camera/rgb/image_raw", Image, detections_pub)
    rospy.spin()

if __name__ == '__main__':
    detections_sub()
    tf_sess.close()
