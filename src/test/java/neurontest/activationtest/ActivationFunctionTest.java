package neurontest.activationtest;

import neuron.activation.ActivationFunction;
import neuron.activation.ReluFunction;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class ActivationFunctionTest {
    private static ActivationFunction relu;

    @BeforeAll
    public static void setUp(){
        relu = new ReluFunction();
    }

    @Test
    public void testActivation(){
        Assertions.assertEquals(3, relu.getActivation(3.0));
        Assertions.assertEquals(0, relu.getActivation(-1.0));
    }

    @Test
    public void testDerivative(){
        Assertions.assertEquals(1, relu.getDerivative(3.0));
        Assertions.assertEquals(0, relu.getDerivative(-1.0));
    }
}
