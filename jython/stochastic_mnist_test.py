"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""
from __future__ import with_statement
import os
import csv
import time
import random
import sys
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance, CrossEntropyError
from func.nn.activation import HyperbolicTangentSigmoid
from opt.example import StochasticNeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

#from __future__ import with_statement

INPUT_FILE = os.path.join("..", "src", "opt", "test", "mnist_train_split.txt")
TEST_FILE = os.path.join("..", "src", "opt", "test", "mnist_test_split.txt")
INPUT_LAYER = 784
HIDDEN_LAYER = 64
OUTPUT_LAYER = 10
TRAINING_ITERATIONS = 1000
BATCH_SIZE = 64

def initialize_instances():
    """Read the abalone.txt CSV data into a list of instances."""
    instances = []
    test = []
    with open(INPUT_FILE, 'r') as f:
        content = f.readlines()
    
    for line in content:
        train, label = map(float, line.split(',')[:-10]), map(int, map(float, line.split(',')[-10:]))        
        instance = Instance(train)
        instance.setLabel(Instance(label))
        instances.append(instance)
    
    with open(TEST_FILE, 'r') as f:
        content = f.readlines()
 
    for line in content:
        line = line.split()
        train, label = map(float, line[0].split(',')), map(int, map(float, line[1].split(',')))
        instance = Instance(train)
        instance.setLabel(Instance(label))
        test.append(instance)
    
    random.shuffle(instances)
    train = instances[:int(len(instances)*0.8)]
    cv = instances[int(len(instances)*0.8):]
    return train, cv, test


def train(oa, network, oaName, instances, cv, measure):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    print "\nError results for %s\n---------------------------" % (oaName,)

    for iteration in xrange(TRAINING_ITERATIONS):
        oa.train()

        error = 0.00
        cv_error = 0.00
        shuffle = random.sample(xrange(len(instances)), BATCH_SIZE)
        cv_shuffle = random.sample(xrange(len(cv)), BATCH_SIZE)
        for i, j in zip(shuffle, cv_shuffle):
            network.setInputValues(instances[i].getData())
            network.run()

            output = instances[i].getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values))
            error += measure.value(output, example)
            
            network.setInputValues(cv[j].getData())
            network.run()

            output = cv[j].getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values))
            cv_error += measure.value(output, example)

        print "training: %0.03f" % error
        print "cv: %0.03f" % cv_error


def main():
    """Run algorithms on the abalone dataset."""
    instances, cv, test = initialize_instances()
    factory = BackPropagationNetworkFactory()
    measure = CrossEntropyError()
    data_set = DataSet(instances)

    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm
    oa_names = ["RHC", "SA", "GA"]
    results = ""

    for name in oa_names:
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER], HyperbolicTangentSigmoid())
        networks.append(classification_network)
        nnop.append(StochasticNeuralNetworkOptimizationProblem(data_set, classification_network, measure, BATCH_SIZE))

    oa.append(RandomizedHillClimbing(nnop[0]))
    oa.append(SimulatedAnnealing(1E11, 0.95, nnop[1]))
    oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[2]))

    for i, name in enumerate(oa_names):
        start = time.time()
        correct = 0
        incorrect = 0

        train(oa[i], networks[i], oa_names[i], instances, cv, measure)
        end = time.time()
        training_time = end - start

        optimal_instance = oa[i].getOptimal()
        networks[i].setWeights(optimal_instance.getData())

        start = time.time()
        for instance in instances:
            networks[i].setInputValues(instance.getData())
            networks[i].run()

            predicted = [index for index in xrange(10) if instance.getLabel().getDiscrete(index) == 1][0]
            actual = networks[i].getOutputValues().get(predicted)

            if actual >= 0.5:
                correct += 1
            else:
                incorrect += 1
        end = time.time()
        testing_time = end - start

        results += "\nResults for Training %s: \nCorrectly classified %d instances." % (name, correct)
        results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (incorrect, float(correct)/(correct+incorrect)*100.0)
        results += "\nTraining time: %0.03f seconds\n" % (training_time,)
        
        start = time.time()
        for instance in cv:
            networks[i].setInputValues(instance.getData())
            networks[i].run()

            predicted = [index for index in xrange(10) if instance.getLabel().getDiscrete(index) == 1][0]
            actual = networks[i].getOutputValues().get(predicted)

            if actual >= 0.5:
                correct += 1
            else:
                incorrect += 1
        end = time.time()
        testing_time = end - start

        results += "\nResults for Cross Validation %s: \nCorrectly classified %d instances." % (name, correct)
        results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%\n" % (incorrect, float(correct)/(correct+incorrect)*100.0)
        
        start = time.time()
        for instance in test:
            networks[i].setInputValues(instance.getData())
            networks[i].run()

            predicted = [index for index in xrange(10) if instance.getLabel().getDiscrete(index) == 1][0]
            actual = networks[i].getOutputValues().get(predicted)

            if actual >= 0.5:
                correct += 1
            else:
                incorrect += 1

        end = time.time()
        testing_time = end - start

        results += "\nResults for Testing %s: \nCorrectly classified %d instances." % (name, correct)
        results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (incorrect, float(correct)/(correct+incorrect)*100.0)
        #results += "\nTraining time: %0.03f seconds" % (training_time,)
        results += "\nTesting time: %0.03f seconds\n" % (testing_time,)
    print results


if __name__ == "__main__":
    main()

