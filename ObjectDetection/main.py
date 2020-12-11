from fasterrcnn import FasterRCNN
from hog import Hog
from yolo import Yolo
import util

from collections import defaultdict
import pandas as pd
from time import time


def time_detection(algos, file):
    times = defaultdict(list)
    for algo in algos:
        for _ in range(100):
            objects, duration = algo.object_detection_api(f'./../src/imgs/{file}')
            times[algo.__name__].append(duration)
        else:
            print(f'{algo.__name__} done')
    return times


if __name__ == '__main__':
    # util.get_system_info()
    files = ['one.jpg', 'two.jpg', 'three.webp', 'four.webp', 'five.png', 'six.jpg']
    algos = [FasterRCNN(), Hog(), Yolo()]

    for file in files:
        times = time_detection(algos, file)
        for name, data in times.items():
            df = pd.DataFrame(data, columns=['Data'])
            util.calc_stats(df, name)
        # util.show_boxplot(times, file.split('.')[0].capitalize())
        util.save_to_file(times, file.split('.')[0])
