package optimiser.loss;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MSEFunction implements LossFunction{
    @Override
    public double getLoss(List<Double> yHat, List<Double> y) {
        return IntStream.range(0, yHat.size())
                .mapToDouble(i ->
                        Math.pow((yHat.get(i) - y.get(i)), 2))
                .sum()/yHat.size();
    }

    @Override
    public List<Double> getDerivative(List<Double> yHat, List<Double> y) {
        return IntStream.range(0, yHat.size())
                .mapToDouble(i -> yHat.get(i) - y.get(i))
                .boxed().collect(Collectors.toList());
    }
}
