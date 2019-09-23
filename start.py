import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
#from face_utils import judge_side_face
from tracker.utils import Logger, mkdir

from tracker.sort import Sort

#from utils import label_map_util

from project_root_dir import project_dir

#from utils import visualization_utils_color as vis_util

from face_detector import FaceDetector

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logger = Logger()


def main():
    global colours, img_size
    args = parse_args()
    root_dir = args.root_dir
    output_path = args.output_path
    display = args.display
    mkdir(output_path)
    # for disp
    if display:
        colours = np.random.rand(32, 3)

    # init tracker
    tracker = Sort()  # create instance of the SORT tracker

    logger.info('Start track and extract......')

    #sys.path.append("..")

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = './model/facebox.pb'

    face_detector = FaceDetector(PATH_TO_CKPT, gpu_memory_fraction=0.25, visible_device_list='0')

    out = None

    # detection_graph = tf.Graph()

    # with detection_graph.as_default():
    #     od_graph_def = tf.GraphDef()
    #     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    #         serialized_graph = fid.read()
    #         od_graph_def.ParseFromString(serialized_graph)
    #         tf.import_graph_def(od_graph_def, name='')


    # with detection_graph.as_default():
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     with tf.Session(graph=detection_graph, config=config) as sess:

    margin = 20  # if the face is big in your video ,you can set it bigger for tracking easiler
            
    frame_interval = 1  # interval how many frames to make a detection,you need to keep a balance between performance and fluency
    scale_rate = 0.9  # if set it smaller will make input frames smaller
    show_rate = 0.8  # if set it smaller will dispaly smaller frames

    for filename in os.listdir(root_dir):
        video_name = os.path.join(root_dir, filename)
        directoryname = os.path.join(output_path, filename.split('.')[0])
        logger.info('Video_name:{}'.format(video_name))
        cam = cv2.VideoCapture(video_name)
        c = 0
        while True:
            final_faces = []
            addtional_attribute_list = []
            ret, frame = cam.read()
            if not ret:
                logger.warning("ret false")
                break
            if frame is None:
                logger.warning("frame drop")
                break

            frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
            r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if c % frame_interval == 0:
                img_size = np.asarray(frame.shape)[0:2]
                #faces, scores, classes, num_detections = detect_face.detect_face_by_frame(detection_graph, sess, r_g_b_frame)
                faces, scores = face_detector(r_g_b_frame, score_threshold=0.3)
                # faces = np.squeeze(faces)
                # scores = np.squeeze(scores)

                face_sums = faces.shape[0]
                        
                if face_sums > 0:
                    face_list = []
                    for i, item in enumerate(faces):
                        item[0],item[1] = item[1],item[0]
                        item[2],item[3] = item[3],item[2]

                        if scores[i] > 0.7:
                            logger.info('detect of a frame: ' + str(item) + '  frame ' + str(c))
                            #det = np.squeeze(faces[i, 0:4])
                            det = np.squeeze(item[0:4])

                            # face rectangle
                            det[0] = np.maximum(det[0] - margin, 0)
                            det[1] = np.maximum(det[1] - margin, 0)
                            det[2] = np.minimum(det[2] + margin, img_size[1])
                            det[3] = np.minimum(det[3] + margin, img_size[0])
                            #item = np.hstack([item, scores[i]])
                            face_list.append(item)

                            # face cropped
                            bb = np.array(det, dtype=np.int32)
                            frame_copy = frame.copy()
                            cropped = frame_copy[bb[1]:bb[3], bb[0]:bb[2], :]

                            # use 5 face landmarks  to judge the face is front or side
                            # squeeze_points = np.squeeze(points[:, i])
                            # tolist = squeeze_points.tolist()
                            # facial_landmarks = []
                            # for j in range(5):
                            #     item = [tolist[j], tolist[(j + 5)]]
                            #     facial_landmarks.append(item)
                            # if args.face_landmarks:
                            #     for (x, y) in facial_landmarks:
                            #         cv2.circle(frame_copy, (int(x), int(y)), 3, (0, 255, 0), -1)
                            # dist_rate, high_ratio_variance, width_rate = judge_side_face(
                            #     np.array(facial_landmarks))
                            facial_landmarks = np.ones((5,2))
                            dist_rate, high_ratio_variance, width_rate = _judge_side_face(facial_landmarks)

                            # face addtional attribute(index 0:face score; index 1:0 represents front face and 1 for side face )
                            item_list = [cropped, scores[i], dist_rate, high_ratio_variance, width_rate]
                            addtional_attribute_list.append(item_list)
                    final_faces = np.array(face_list)

            trackers = tracker.update(final_faces, img_size, directoryname, addtional_attribute_list, r_g_b_frame)

            c += 1

            for d in trackers:
                if display:
                    d = d.astype(np.int32)
                    cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 5)
                    cv2.putText(frame, 'ID : %d' % (d[4]), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                colours[d[4] % 32, :] * 255, 2)
                    
            # write every frame to video
            if out is None:
                [h, w] = frame.shape[:2]
                out = cv2.VideoWriter(os.path.join(root_dir, filename.split('.')[0] + "_track_out.avi"), 0, 25.0, (w, h))
            out.write(frame)
                    
    cam.release()
    out.release()

def _judge_side_face(facial_landmarks):
    dist_rate = .5
    high_ratio_variance = .5
    width_rate = .5
    return dist_rate, high_ratio_variance, width_rate

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str,
                        help='Path to the data directory containing aligned your face patches.', default='./media')
    parser.add_argument('--output_path', type=str,
                        help='Path to save face',
                        default='facepics')
    parser.add_argument('--display', type=bool,
                        help='Display or not', default=True)
    parser.add_argument('--face_landmarks', type=bool,
                        help='draw 5 face landmarks on extracted face or not ', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
        


    