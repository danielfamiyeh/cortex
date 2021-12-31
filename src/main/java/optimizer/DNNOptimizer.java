package optimizer;

import neuron.Layer;
import neuron.Neuron;
import optimizer.loss.LossFunction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.BiConsumer;

public class DNNOptimizer {
  public DNNOptimizer() {
  }

  public List<List<Double>> optimise(List<Layer> network,
                                 List<List<Double>> dataset,
                       List<List<Double>> labels, LossFunction lossFunction,
                       int numEpochs, double alpha, BiConsumer<Neuron, Double> algorithm) {

    int numExamples = dataset.size();
    Layer inputLayer = network.get(0);
    Layer outputLayer = network.get(network.size() - 1);
    List<Double> errorVector;
    List<Double> yHat;
    List<List<Double>> finalError = new ArrayList<>();

    for (int epoch = 0; epoch < numEpochs; epoch++) {
      double cost = 0.0;

      for (int i = 0; i < numExamples; i++) {
        List<Double> feature = dataset.get(i);
        inputLayer.setActivation(feature);
        for (Layer layer : network) {
          layer.forward();
        }
        yHat = outputLayer.getActivation();
        cost += lossFunction.getLoss(yHat,
            labels.get(i));

        errorVector = lossFunction.getDerivative(yHat, labels.get(i));

        for (int j = 0; j < errorVector.size(); j++) {
          Neuron outputNeuron = outputLayer.getNeuronList().get(j);
          errorVector.set(j, errorVector.get(j) *
              outputNeuron.getFunction().getDerivative(
                  outputNeuron.getNetInput()));
        }

        if (epoch == numEpochs - 1) {
          finalError.add(new ArrayList<>(errorVector));
        }
        // Backprop
        outputLayer.setError(errorVector);
        for (int j = network.size() - 2; j > 0; j--) {
          Layer hiddenLayer = network.get(j);
          for (Neuron neuron : hiddenLayer.getNeuronList()) {
            double neuronError = neuron.getOutputAxons()
                .stream()
                .mapToDouble(axon ->
                    axon.getWeight()
                        * axon.getDest().getError())
                .sum() / neuron.getOutputAxons().size();
            neuronError *= neuron.getFunction().getDerivative(
                neuron.getNetInput());
            neuron.setError(neuronError);
          }
        }
        // Weight update
        for (int j = network.size() - 1; j > 0; j--) {
          network.get(j).updateWeights(algorithm, alpha);
        }
      }
      cost /= numExamples;
      //        System.out.println("Epoch: " + (epoch+1));
      //      System.out.println("Average Error: " + cost + "\n");
    }

    return finalError;
  }
}
