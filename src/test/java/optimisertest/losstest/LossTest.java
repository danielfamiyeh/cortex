package optimisertest.losstest;

import optimiser.loss.LossFunction;
import optimiser.loss.MSEFunction;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class LossTest {
    private static List<Double> yHat;
    private static List<Double> y;
    private static LossFunction mse;

    @BeforeAll
    public static void setUp(){
        yHat = Arrays.asList(0.23, 0.45);
        y = Arrays.asList(1.0, 0.0);
        mse = new MSEFunction();
    }

    @Test
    public void testLoss(){
        // MSE output test
        Assertions.assertEquals(
                IntStream.range(0, yHat.size())
                .mapToDouble(i ->
                        Math.pow(yHat.get(i) - y.get(i), 2))
                .sum()/yHat.size(),
                mse.getLoss(yHat, y));
    }

    @Test
    public void testDerivative(){
        // MSE derivative test
        Assertions.assertEquals(
          IntStream.range(0, yHat.size())
          .mapToDouble(i ->
                  yHat.get(i) - y.get(i))
                  .boxed()
                  .collect(Collectors.toList()),

                mse.getDerivative(yHat, y)
        );
    }
}
