import csv
import random
import numpy as np

from RBM import RBM

def load_dataset():
    file = csv.DictReader(open('iris.csv'))
    lines = [line for line in file]

    features = ['len_cup', 'width_cup', 'len_petal', 'width_petal']
    maxes = {}
    mins = {}
    for feature in features:
        maxes[feature] = float(max(lines, key=lambda l: float(l[feature]))[feature])
        mins[feature] = float(min(lines, key=lambda l: float(l[feature]))[feature])

    visible_neurons_dataset = []
    for line in lines:
        v = []

        # add 3 neurons for each feature
        for feature in features:
            feature_val = float(line[feature])
            third_length = (maxes[feature] - mins[feature]) / 3.0
            lower_third = mins[feature] + third_length
            upper_third = lower_third + third_length

            v.append(int(feature_val < lower_third))
            v.append(int(lower_third <= feature_val < upper_third))
            v.append(int(upper_third <= feature_val))

        # add 3 classification neurons - one for each option
        v.append(int(line['type'] == '0'))
        v.append(int(line['type'] == '1'))
        v.append(int(line['type'] == '2'))

        v = np.array(v)

        visible_neurons_dataset.append(v)

    return visible_neurons_dataset

def main():
    nv = 15 # 3 for each feature (4 features) + 3 for iris type
    nh = 20
    in_mask = [1] * 12 + [0] * 3
    rbm = RBM(nv, nh)
    training_set = load_dataset()
    v = random.choice(training_set)
    print(v)
    vr = rbm.infer(v, in_mask, 4)
    print(vr)

    rbm.train(training_set, 1000)
    vr = rbm.infer(v, in_mask, 4)
    print(vr)

if __name__ == '__main__':
    main()
