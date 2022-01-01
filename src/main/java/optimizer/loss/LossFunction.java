package optimizer.loss;

import java.util.List;

/**
 * Interface representing a loss function
 */
public interface LossFunction {

  /**
   * Calculates loss based on expected and actual outcomes
   * @param y     Actual outcome
   * @param yHat  Expected outcome
   * @return      Loss value
   */
  double getLoss(List<Double> y,
                 List<Double> yHat);

  /**
   * Calculates derivative of loss based on expected and actual outcomes
   * @param y     Actual outcome
   * @param yHat  Expected outcome
   * @return      Derivative of loss
   */
  List<Double> getDerivative(List<Double> y,
                             List<Double> yHat);
}
