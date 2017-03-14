package shared;
import java.lang.Math;



/**
 * Cross Entropy measure, suitable for use with
 * soft max networks for multi class probabilities.
 * @author Chunyan BAI cbai32@gatech.edu
 * @version 1.0
 */
public class CrossEntropyError extends AbstractErrorMeasure
        implements GradientErrorMeasure {

    /**
     * @see nn.error.ErrorMeasure#error(double[], nn.Pattern[], int)
     */
    public double value(Instance output, Instance example) {
        double sum = 0;
        Instance label = example.getLabel();
        for (int i = 0; i < output.size(); i++) {
            sum += output.getContinuous(i) * Math.log(label.getContinuous(i));
        }
        return -1 * sum;
    }

    /**
     * @see nn.error.DifferentiableErrorMeasure#derivatives(double[], nn.Pattern[], int)
     */
    public double[] gradient(Instance output, Instance example) {      
        double[] errorArray = new double[output.size()];
        Instance label = example.getLabel();
        for (int i = 0; i < output.size(); i++) {
            errorArray[i] = -1 * output.getContinuous(i) / label.getContinuous(i); 
        }
        return errorArray;
    }

}
