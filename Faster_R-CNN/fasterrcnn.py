from collections import defaultdict
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import subprocess
from time import time
import torchvision
import torchvision.transforms as T


def get_system_info():
    # traverse the info
    info = subprocess.check_output(['systeminfo']).decode('utf-8').split('\n')
    new = []

    # arrange the string into clear info
    for item in info:
        new.append(str(item.split("\r")[:-1]))

    for i in new:
        print(i[2:-2])


class ObjectDetection:
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

    def get_prediction(self, img_path):
        img = Image.open(img_path)  # Load the image
        transform = T.Compose([T.ToTensor()])  # Defining PyTorch Transform
        img = transform(img)  # Apply the transform to the image
        pred = self.model([img])  # Pass the image to the model
        return pred[0]

    def object_detection_api(self, img_path, threshold=0.9, rect_th=2, text_size=1, text_th=2, show_img=False):
        found = []

        img = cv2.imread(img_path)  # Read image with cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        start = time()
        pred = self.get_prediction(img_path)  # Get predictions
        end = time()

        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            top_left = (float(box[0]), float(box[1]))
            bottom_right = (float(box[2]), float(box[3]))
            box = (top_left, bottom_right)
            label = ObjectDetection.COCO_INSTANCE_CATEGORY_NAMES[int(label)]
            score = float(score)
            if label == 'person' and score >= threshold:
                found.append((label, score, box))
        #           print(box,label,score)

        if show_img:
            for label, score, box in found:
                x_min, y_min = map(int, box[0])
                x_max, y_max = map(int, box[1])
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0, 255, 0),
                              thickness=rect_th)  # Draw Rectangle with the coordinates
                cv2.putText(img, label, (int(x_min + 5 * text_size), int(y_min + 30 * text_size)),
                            cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 0, 0),
                            thickness=text_th)  # Write the prediction class
                cv2.putText(img, f'{round(score * 100)}%', (int(x_min + 5 * text_size), int(y_min + 70 * text_size)),
                            cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 0, 0), thickness=text_th)

            plt.figure(figsize=(20, 30))  # display the output image
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.show()

        sec = end - start
        print(f'Object Detection took {round(sec, 3)} seconds on image size {img.shape}')
        return found, sec

    def time_detection(self, files, timeout):
        times = defaultdict(list)
        for file in files:
            start = time()
            while time() - start < timeout:
                objects, duration = self.object_detection_api(f'./../src/imgs/{file}')
                times[file.capitalize()].append(duration)
            else:
                print(f'Image {file} done')
        return times

    def calc_stats(self, dataset, dataname):
        mean = float(np.round(np.mean(dataset), 3))
        median = float(np.round(np.median(dataset), 3))
        min_value = float(np.round(dataset.min(), 3))
        max_value = float(np.round(dataset.max(), 3))
        quartile_1 = float(np.round(dataset.quantile(0.25), 3))
        quartile_3 = float(np.round(dataset.quantile(0.75), 3))
        iqr = np.round(quartile_3 - quartile_1, 3)
        lower_bound = np.round(quartile_1 - iqr * 1.5, 3)
        upper_bound = np.round(quartile_3 + iqr * 1.5, 3)

        print(f'{dataname} summary statistics')
        print(f'Min                      : {min_value}')
        print(f'Mean                     : {mean}')
        print(f'Max                      : {max_value}')
        print('')
        print(f'25th percentile          : {quartile_1}')
        print(f'Median                   : {median}')
        print(f'75th percentile          : {quartile_3}')
        print(f'Interquartile range (IQR): {iqr}')
        print('')
        print(f'Lower outlier bound      : {lower_bound}')
        print(f'Upper outlier bound      : {upper_bound}')
        print('--------------------------------')

    def show_boxplot(self, times):
        # Create a figure instance
        fig = plt.figure(1, figsize=(9, 6))
        # Create an axes instance
        ax = fig.add_subplot(111)
        # Create the boxplot
        bp = ax.boxplot(times.values())
        # add patch_artist=True option to ax.boxplot() to get fill color
        bp = ax.boxplot(times.values(), patch_artist=True)
        # Custom x-axis labels
        ax.set_xticklabels(times.keys())
        # Remove top axes and right axes ticks
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # Show the figure
        plt.show()


if __name__ == '__main__':
    get_system_info()
    od = ObjectDetection()
    # timeout = 60*10 # 10 minutes
    timeout = 60  # 1 minutes
    files = ['one.jpg', 'two.jpg', 'three.webp', 'four.webp', 'five.png', 'six.jpg']
    times = od.time_detection(files, timeout)
    for name, data in times.items():
        df = pd.DataFrame(data, columns=['Data'])
        od.calc_stats(df, name)
    od.show_boxplot(times)
