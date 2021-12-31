package optimiser.loss;

import java.util.List;

public interface LossFunction {
  double getLoss(List<Double> yHat,
                 List<Double> y);

  List<Double> getDerivative(List<Double> yHat,
                             List<Double> y);
}
