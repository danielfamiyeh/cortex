package optimizer.algorithm;

import neuron.Layer;
import optimizer.loss.LossFunction;

import java.util.List;

public interface Optimiser {
  void optimiseDNN(List<Layer> network, List<List<Double>> dataset,
                   List<List<Double>> labels, LossFunction lossFunction, int numEpochs, double alpha);
}
