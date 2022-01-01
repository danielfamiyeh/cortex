package neuron;

import neuron.activation.ActivationFunction;

import java.util.List;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Class representing a 1D layer of neurons
 */
public class Layer {
  private List<Double> activation;
  private List<Neuron> neuronList;
  private ActivationFunction function;

  /**
   *
   * @param size      Number of neurons in layer
   * @param function  Activation function
   */
  public Layer(int size, ActivationFunction function) {
    this.function = function;
    activation = IntStream.range(0, size)
        .mapToDouble(i -> 0.0)
        .boxed().collect(Collectors.toList());
    neuronList = IntStream.range(0, size)
        .mapToObj(i -> new Neuron(function))
        .collect(Collectors.toList());
  }

  /**
   * Feedforward algorithm
   */
  public void forward() {
    if (function != null) {
      for (int neuron = 0; neuron < neuronList.size(); neuron++) {
        activation.set(neuron, neuronList.get(neuron).forward());
      }
    }
  }

  /**
   * Connect with a single neuron of another layer
   * @param dest          Destination layer
   * @param sourceIndex   Source neuron index
   * @param destIndex     Destination neuron index
   * @param weight        Weight of axon connection
   */
  public void connect(
      Layer dest, int sourceIndex,
      int destIndex, double weight) {
    try {
      Neuron sourceNeuron = neuronList.get(sourceIndex);
      Neuron destNeuron = dest.neuronList.get(destIndex);

      Axon sourceToDest = new Axon(weight, destNeuron);
      Axon destToSource = new Axon(weight, sourceNeuron);

      sourceNeuron.addOutputAxon(sourceToDest);
      destNeuron.addInputAxon(destToSource);

    } catch (ArrayIndexOutOfBoundsException aioobe) {
      System.out.println("Invalid index passed to Layer.connect() method");
    }
  }

  /**
   * Fully-connect with another layer
   * @param dest
   * @param weight
   */
  public void connect(Layer dest, double weight) {
    for (int sourceIndex = 0;
         sourceIndex < neuronList.size();
         sourceIndex++) {
      for (int destIndex = 0;
           destIndex < dest.neuronList.size();
           destIndex++) {
        connect(dest, sourceIndex, destIndex, weight);
      }
    }
  }

  public void resetDeltas() {
    neuronList.forEach(Neuron::resetDeltas);
  }

  /**
   * Disconnect all neurons from their sources/destinations
   */
  public void disconnect() {
    neuronList.forEach(Neuron::disconnect);
  }

  /**
   * Fully-connect with another layer automating weight
   * @param dest
   */
  public void connect(Layer dest) {
    connect(dest, (this.function == null) ? 1 : Math.random());
  }

  /**
   * Get output layer's activation
   * @return List of activations
   */
  public List<Double> getActivation() {
    return activation;
  }

  /**
   * Set activation of output layer
   * @param a List of activation values
   */
  public void setActivation(List<Double> a) {
    for (int i = 0; i < neuronList.size(); i++) {
      neuronList.get(i).setActivation(a.get(i));
      activation.set(i, a.get(i));
    }
  }

  /**
   * Updates weights for all neurons in layer
   * @param updateRoutine Algorithm used for weight updates
   * @param alpha         Learning rate
   */
  public void updateWeights(BiConsumer<Neuron, Double> updateRoutine, double alpha) {
    for (Neuron neuron : neuronList) updateRoutine.accept(neuron, alpha);
  }

  /**
   * Get number of neurons in layer
   * @return Number of neurons in layer
   */
  public int getSize() {
    return neuronList.size();
  }

  /**
   * Set the bias of each neuron to a single value
   * @param b Bias value
   */
  public void setBias(double b) {
    neuronList.forEach(neuron -> neuron.setBias(b));
  }

  /**
   * Set the loss of every neuron individually
   * @param e List of loss values
   */
  public void setError(List<Double> e) {
    for (int i = 0; i < neuronList.size(); i++) {
      neuronList.get(i).setError(e.get(i));
    }
  }

  /**
   * Get list of neuron objects in layer
   * @return List of neuron objects
   */
  public List<Neuron> getNeuronList() {
    return neuronList;
  }
}
