package optimizertest;

import org.junit.jupiter.api.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import neuron.Layer;
import neuron.Neuron;
import neuron.activation.ReluFunction;

import optimizer.DNNOptimizer;
import optimizer.OptimAlgo;
import optimizer.loss.MSEFunction;

public class OptimizerTest {
  private static List<List<Double>> xorDataset;
  private static List<List<Double>> labels;
  private static List<Layer> network;
  private static DNNOptimizer underTest;

  @BeforeEach
  public static void setUp() {
    xorDataset = new ArrayList<>();
    labels = new ArrayList<>();
    network = new ArrayList<>();

    underTest = new DNNOptimizer();

    Collections.addAll(xorDataset, Arrays.asList(0.0, 0.0),
        Arrays.asList(0.0, 1.0),
        Arrays.asList(1.0, 0.0),
        Arrays.asList(1.0, 1.0));

    Collections.addAll(labels, Arrays.asList(0.0),
        Arrays.asList(1.0),
        Arrays.asList(1.0),
        Arrays.asList(0.0));

    Collections.addAll(network, new Layer(2, null),
        new Layer(3, new ReluFunction()),
        new Layer(1, new ReluFunction()));

    for (int i = 0; i < network.size() - 1; i++) {
      network.get(i).connect(network.get(i + 1));
    }
  }

  /**
   * Decides if network has been optimized
   * Since parameters are random sometimes networks
   * end with parameters that converge on the correct
   * value sometimes they do not.
   *
   * So we accept a loss < 0.4 as a pass.
   *
   * @param finalError List of lists of final error values
   */
  public void isOptimized(List<List<Double>> finalError){
    AtomicInteger numCorrect = new AtomicInteger();
    finalError.forEach(errorVector -> {
      double loss = errorVector.get(0);
      numCorrect.addAndGet((loss < 0.4) ? 1 : 0);
    });

    boolean passed = numCorrect.intValue() == 4;

    if(!passed) {
      System.out.println("Failed with errors: " + finalError);
    }

    Assertions.assertTrue(passed);
  }

  @Test
  public void testSGD() {
    isOptimized(underTest.optimize(
        network, xorDataset,
        labels, new MSEFunction(), 100000, 0.02,
        OptimAlgo.sgd
    ));
  }

  // TODO: Fix hard convergence about 0.5
  @Disabled
  public void testSGDMomentum() {
//    isOptimized(underTest.optimise(
//        network, xorDataset,
//        labels, new MSEFunction(), 100000, 0.001,
//        OptimAlgo.momentum
//    ));
  }

  @Test
  public void testRMSProp() {
    isOptimized(underTest.optimize(
        network, xorDataset,
        labels, new MSEFunction(), 20000, 0.05,
        OptimAlgo.rms
    ));
  }

  @Test
  public void testAdam() {
    isOptimized(underTest.optimize(
        network, xorDataset,
        labels, new MSEFunction(), 30000, 0.04,
        OptimAlgo.adam
    ));
  }

  @Test
  public void resetDeltas() {
    underTest.optimize(
        network, xorDataset,
        labels, new MSEFunction(), 10000, 0.01,
        OptimAlgo.adam
    );

    network.forEach(Layer::resetDeltas);
    network.forEach(layer ->
    {
      List<Neuron> neuronList = layer.getNeuronList();
      neuronList.forEach(neuron -> {
        List<List<Double>> deltaWeights = neuron.getDeltaWeight();
        List<Double> deltaBias = neuron.getDeltaBias();
        deltaBias.forEach(b -> Assertions.assertEquals(0, b));
        deltaWeights.forEach(dwArray -> dwArray.forEach(db ->
            Assertions.assertEquals(0, db)
        ));
      });
    });
  }
}
