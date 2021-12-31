package neurontest;

import neuron.Axon;
import neuron.Neuron;
import neuron.activation.ReluFunction;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

public class NeuronTest {
  private static Neuron underTest;
  private static Neuron dest1;
  private static Neuron dest2;
  private static Axon axon1;
  private static Axon axon2;
  private static double activation;
  private static double netInput;

  @BeforeEach
  public void setUp() {
    underTest = new Neuron(new ReluFunction());
    dest1 = new Neuron(null);
    dest2 = new Neuron(null);

    axon1 = new Axon(1.7, dest1);
    axon2 = new Axon(-2.3, dest2);

    netInput = (underTest.getBias() - 1.2) / 2;
    activation = 0;
  }

  @Test
  public void testAddInputAxon() {
    underTest.addInputAxon(axon1);
    Assertions.assertEquals(axon1, underTest.getInputAxons().get(0));
    underTest.addInputAxon(axon2);
    Assertions.assertEquals(axon2, underTest.getInputAxons().get(1));
  }

  @Test
  public void testAddOutputAxon() {
    underTest.addOutputAxon(axon2);
    Assertions.assertEquals(axon2, underTest.getOutputAxons().get(0));
    underTest.addOutputAxon(axon1);
    Assertions.assertEquals(axon1, underTest.getOutputAxons().get(1));
  }

  @Test
  public void testRandomiseInputWeights() {
    double[] weights = {1.7, -2.3};

    underTest.addInputAxon(axon1);
    underTest.addInputAxon(axon2);
    underTest.randomiseInputWeights();

    for (int i = 0; i < weights.length; i++) {
      List<Axon> inputAxons = underTest.getInputAxons();
      double weight = inputAxons.get(i).getWeight();
      Assertions.assertTrue(weights[i] != weight &&
          weight >= -0.5 && weight <= 0.5);
    }
  }

  @Test
  public void testRandomiseOutputWeights() {
    double[] weights = {-2.3, 1.7};

    underTest.addOutputAxon(axon2);
    underTest.addOutputAxon(axon1);
    underTest.randomiseOutputWeights();

    for (int i = 0; i < weights.length; i++) {
      List<Axon> outputAxons = underTest.getOutputAxons();
      double weight = outputAxons.get(i).getWeight();
      Assertions.assertTrue(weights[i] != weight &&
          weight >= -0.5 && weight <= 0.5);
    }
  }

  @Test
  public void testGetBias() {
    double bias = underTest.getBias();
    Assertions.assertTrue(bias >= -0.5 &&
        bias <= 0.5);
  }

  @Test
  public void testSetBias() {
    underTest.setBias(0.87);
    Assertions.assertEquals(0.87, underTest.getBias());
  }

  @Test
  public void testGetActivation() {
    Assertions.assertEquals(0, underTest.getActivation());
  }

  @Test
  public void testSetActivation() {
    underTest.setActivation(2.0);
    Assertions.assertEquals(2.0, underTest.getActivation());
  }

  @Test
  public void testForward() {
    dest1.setActivation(2.0);
    dest2.setActivation(2.0);

    underTest.addInputAxon(axon1);
    underTest.addInputAxon(axon2);

    underTest.forward();

    dest1.setActivation(4);
    underTest.forward();

    double ws = underTest.getNetInput() * 2 - underTest.getBias();
    double epsilon = Math.abs(ws - 2.2);

    Assertions.assertTrue(epsilon < 1);
  }
}
