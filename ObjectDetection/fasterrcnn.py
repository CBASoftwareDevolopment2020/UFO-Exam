import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from time import time
import torchvision
import torchvision.transforms as T


class FasterRCNN:
    __name__ = 'FasterRCNN'
    COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                                    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                                    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
                                    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                                    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                                    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                                    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                                    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
                                    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
                                    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        return model

    def get_prediction(self, img, transform):
        img = transform(img)  # Apply the transform to the image
        pred = self.model([img])  # Pass the image to the model
        return pred[0]

    def object_detection_api(self, img_path, threshold=0.9, rect_th=2, text_size=1, text_th=2, show_img=False):
        pil_img = Image.open(img_path)  # Load the image
        transform = T.Compose([T.ToTensor()])  # Defining PyTorch Transform
        start = time()
        pred = self.get_prediction(pil_img, transform)  # Get predictions
        end = time()

        found = []
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            top_left = (float(box[0]), float(box[1]))
            bottom_right = (float(box[2]), float(box[3]))
            box = (top_left, bottom_right)
            label = FasterRCNN.COCO_INSTANCE_CATEGORY_NAMES[int(label)]
            score = float(score)
            if label == 'person' and score >= threshold:
                found.append((label, score, box))
        #           print(box,label,score)

        cv2_img = cv2.imread(img_path)  # Read image with cv2
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        if show_img:
            for label, score, box in found:
                x_min, y_min = map(int, box[0])
                x_max, y_max = map(int, box[1])
                cv2.rectangle(cv2_img, (x_min, y_min), (x_max, y_max), color=(0, 255, 0),
                              thickness=rect_th)  # Draw Rectangle with the coordinates
                cv2.putText(cv2_img, label, (int(x_min + 5 * text_size), int(y_min + 30 * text_size)),
                            cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 0, 0),
                            thickness=text_th)  # Write the prediction class
                cv2.putText(cv2_img, f'{round(score * 100)}%',
                            (int(x_min + 5 * text_size), int(y_min + 70 * text_size)),
                            cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 0, 0), thickness=text_th)

            plt.figure(figsize=(20, 30))  # display the output image
            plt.imshow(cv2_img)
            plt.xticks([])
            plt.yticks([])
            plt.show()

        sec = end - start
        print(f'Object Detection took {round(sec, 3)} seconds on image size {cv2_img.shape}')
        return found, sec


if __name__ == '__main__':
    od = FasterRCNN()
    # timeout = 60*10 # 10 minutes
    timeout = 60  # 1 minutes
    files = ['one.jpg', 'two.jpg', 'three.webp', 'four.webp', 'five.png', 'six.jpg']
    times = od.time_detection(files, timeout)
    for name, data in times.items():
        df = pd.DataFrame(data, columns=['Data'])
        od.calc_stats(df, name)
    od.show_boxplot(times, 'boxplot')
