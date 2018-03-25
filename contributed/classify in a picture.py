# coding=utf-8
import face
import argparse
import cv2
#from scipy import misc
#import matplotlib.pyplot as plt
import sys

def main(args):
    face_recognition = face.Recognition()
    img = cv2.imread(args.img_path)
    #img = cv2.resize(img,(500,700))
    cv2.imshow('orignal',img)
    human_faces =face_recognition.identify(img)
    for human_face in human_faces:
        print(human_face.name)
        face_bb = human_face.bounding_box.astype(int)
        cv2.rectangle(img,
                      (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                      (0, 255, 0), 2)
        cv2.putText(img, human_face.name, (face_bb[0], face_bb[3]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    thickness=2, lineType=2)
        cv2.imshow('recognized',img)

        k = cv2.waitKey(0)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str,
                        help='Path to the picture directory', default='E:\code_for_graduation\human_picture/picture1.jpg')
    return parser.parse_args(argv)
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))