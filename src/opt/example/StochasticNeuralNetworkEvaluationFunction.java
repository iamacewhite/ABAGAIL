package opt.example;

import util.linalg.Vector;
import func.nn.NeuralNetwork;
import opt.EvaluationFunction;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import java.util.*;

/**
 * An stochastic evaluation function that uses a neural network for performance
 * @author Chunyan BAI
 * @version 1.0
 */
public class StochasticNeuralNetworkEvaluationFunction implements EvaluationFunction {
    /**
     * The network
     */
    private NeuralNetwork network;
    /**
     * The examples
     */
    private DataSet examples;
    /**
     * The error measure
     */
    private ErrorMeasure measure;
    /**
     * All indicies for stochastic update
     */
    private ArrayList<Integer> indicies;
    /**
     * Batch Size
     */
    private int batchSize;
    /**
     * Make a new neural network evaluation function
     * @param network the network
     * @param examples the examples
     * @param measure the error measure
     */
    public StochasticNeuralNetworkEvaluationFunction(NeuralNetwork network,
            DataSet examples, ErrorMeasure measure, int batchSize) {
        this.network = network;
        this.examples = examples;
        this.measure = measure;
        this.batchSize = batchSize;
        this.indicies = new ArrayList<Integer>();
        for (int i = 0; i < examples.size(); i++) {
            this.indicies.add(i);
        }
    }

    /**
     * @see opt.OptimizationProblem#value(opt.OptimizationData)
     */
    public double value(Instance d) {
        // set the links
        Vector weights = d.getData();
        network.setWeights(weights);
        // calculate the error
        double error = 0;
        Collections.shuffle(indicies);
        for (int i = 0; i < batchSize; i++) {
            network.setInputValues(examples.get(indicies.get(i)).getData());
            network.run();
            error += measure.value(new Instance(network.getOutputValues()), examples.get(indicies.get(i)));
        }
        // the fitness is 1 / error
        return 1 / error;
    }

}
