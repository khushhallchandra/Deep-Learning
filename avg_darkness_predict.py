from collections import defaultdict
#defaultdict is superior than dict
#if a key is not there it simply adds that key to the dict
import cPickle
import gzip
import numpy as np

def main():
    training_data, validation_data, test_data = load()
    avgs = avg_darknesses(training_data)
    num_correct = sum(int(guess_digit(image, avgs) == digit) for image, digit in zip(test_data[0], test_data[1]))
    print "Baseline classifier using average darkness of image."
    print "%s of %s values correct." % (num_correct, len(test_data[1]))

def avg_darknesses(training_data):
    digit_counts = defaultdict(int)
    darknesses = defaultdict(float)
    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1
        darknesses[digit] += sum(image)
    avgs = defaultdict(float)
    for digit, n in digit_counts.iteritems():
        avgs[digit] = darknesses[digit] / n
    return avgs

def guess_digit(image, avgs):
    darkness = sum(image)
    distances = {k: abs(v-darkness) for k, v in avgs.iteritems()}
    return min(distances, key=distances.get)
#load the dataset
def load():
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

if __name__ == "__main__":
    main()