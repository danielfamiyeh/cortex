package neurontest;

import neuron.Axon;
import neuron.Neuron;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

/**
 * Axon test suite
 */
public class AxonTest {
  private static Axon underTest;
  private static Neuron testDest;

  @BeforeEach
  public static void setUp() {
    testDest = new Neuron(null);
    underTest = new Axon(7, testDest);
  }

  @Test
  public void getDestTest() {
    Assertions.assertEquals(testDest, underTest.getDest());
  }

  @Test
  public void getWeightTest() {
    Assertions.assertEquals(7, underTest.getWeight());
  }

  @Test
  public void setWeightTest() {
    underTest.setWeight(-10);
    Assertions.assertEquals(-10, underTest.getWeight());
  }

  @Test
  public void decrementWeight() {
    underTest.decrementWeight(1);
    Assertions.assertEquals(6, underTest.getWeight());
  }

  @Test
  public void randomiseWeight() {
    double randomWeight;
    underTest.randomizeWeight();
    randomWeight = underTest.getWeight();

    Assertions.assertTrue(randomWeight >= -0.5 &&
        randomWeight <= 0.5
        && randomWeight != 7);
  }
}
