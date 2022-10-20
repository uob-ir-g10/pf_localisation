#!/usr/bin/python3
import sys
import pandas as pd

from bagpy import bagreader
from matplotlib import pyplot as plt
import math

def main():
    MAP_WIDTH, MAP_HEIGHT = 602, 602
    MAP_RESOLUTION = 0.05
    b = bagreader(sys.argv[1])

    csvfiles = []
    for t in b.topics:
        data = b.message_by_topic(t)
        csvfiles.append(data)
    exact = pd.read_csv(csvfiles[0])
    exact = exact[["header.stamp.secs", "header.stamp.nsecs", "pose.pose.position.x",
                          'pose.pose.position.y', 'pose.pose.orientation.z', 'pose.pose.orientation.w']]
    exact = exact[exact["header.stamp.nsecs"] % 200000000 == 0]
    exact["pose.pose.position.x"] = exact["pose.pose.position.x"] + MAP_WIDTH * MAP_RESOLUTION / 2
    exact["pose.pose.position.y"] = exact["pose.pose.position.y"] + MAP_HEIGHT * MAP_RESOLUTION / 2
    

    estim = pd.read_csv(csvfiles[1])
    estim = estim[['header.stamp.secs', "header.stamp.nsecs", 'pose.position.x', 'pose.position.y', 'pose.orientation.z', 'pose.orientation.w']]

    start_time = max(estim['header.stamp.secs'].iloc[0], exact['header.stamp.secs'].iloc[0])
    estim = estim[estim['header.stamp.secs'] > start_time]
    exact = exact[exact['header.stamp.secs'] > start_time]
    exact.reset_index(inplace=True)
    estim.reset_index(inplace=True)

    exact.to_csv('exact.csv')
    estim.to_csv('estim.csv')
    both = pd.concat([exact, estim], axis=1)
    both.to_csv('both.csv')

    distances = ((exact['pose.pose.position.x'] - estim['pose.position.x']) * (exact['pose.pose.position.x'] - estim['pose.position.x']) + (exact['pose.pose.position.y'] - estim['pose.position.y']) * (exact['pose.pose.position.y'] - estim['pose.position.y'])).tolist()
    distances = list(map(math.sqrt, distances))
    times = (exact['header.stamp.secs'] + 10**(-9) * exact['header.stamp.nsecs']).tolist()
    plt.plot(times, distances)
    plt.show()


if __name__ == "__main__":
    main()