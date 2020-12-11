import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from time import time


class Yolo:
    __name__ = 'Yolo'

    def get_prediction(self, img):
        # ------ TIME THIS BLOCK ------------
        bboxs, labels, confs = cv.detect_common_objects(img)
        return bboxs, labels, confs
        # ----------------------------------

    def object_detection_api(self, fname, show_img=False):
        img = cv2.imread(fname)
        start = time()
        bboxs, labels, confs = self.get_prediction(img)
        end = time()
        pbboxs = []
        plabels = []
        pconfs = []

        for i in range(len(labels)):
            if labels[i] == "person":
                pbboxs.append(bboxs[i])
                plabels.append(labels[i])
                pconfs.append(confs[i])
                cv2.putText(img, str(confs[i]), (bboxs[i][0] + 100, bboxs[i][1] + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 1)
        if show_img:
            output_image = draw_bbox(img, pbboxs, plabels, pconfs)
            # cv2.imshow(output_image)
            # plt.show()
        sec = end - start
        print(f'Object Detection took {round(sec, 3)} seconds on image size {img.shape}')
        return zip(pbboxs, plabels, pconfs), sec
