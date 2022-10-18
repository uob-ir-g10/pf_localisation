#!/usr/bin/python3
from matplotlib import pyplot as plt
from pf_localisation import pf
from numpy import arange

def main():
    particle_filter = pf.PFLocaliser()

    observed = list(arange(0.0, 7.0, 0.01))    
    probs1 = list(map(lambda x: particle_filter.sensor_model.predict(x, 0.5), observed))
    probs2 = list(map(lambda x: particle_filter.sensor_model.predict(x, 3), observed))
    probs3 = list(map(lambda x: particle_filter.sensor_model.predict(x, 6), observed))


    fig, axis = plt.subplots(1, 3)
    axis[0].plot(observed, probs1)
    axis[0].set_title("Wall at x = 0.5")

    axis[1].plot(observed, probs2)
    axis[1].set_title("Wall at x = 3")

    axis[2].plot(observed, probs3)
    axis[2].set_title("Wall at x = 6")
    fig.text(0.5, 0.04, 'Measurement', ha='center', va='center')
    fig.text(0.06, 0.5, 'Probability', ha='center', va='center', rotation='vertical')
    plt.show()

if __name__ == '__main__':
    main()