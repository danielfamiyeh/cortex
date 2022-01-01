package optimizer.loss;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Class representing the mean-squared error loss function
 */
public class MSEFunction implements LossFunction {
  @Override
  public double getLoss(List<Double> y, List<Double> yHat) {
    return IntStream.range(0, y.size())
        .mapToDouble(i ->
            Math.pow((y.get(i) - yHat.get(i)), 2))
        .sum() / y.size();
  }

  @Override
  public List<Double> getDerivative(List<Double> y, List<Double> yHat) {
    return IntStream.range(0, y.size())
        .mapToDouble(i -> y.get(i) - yHat.get(i))
        .boxed().collect(Collectors.toList());
  }
}
