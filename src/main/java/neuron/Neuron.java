package neuron;

import neuron.activation.ActivationFunction;
import neuron.activation.ReluFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Class representing a neuron
 */
public class Neuron {
  double bias;
  double error;
  List<List<Double>> deltaWeight;
  List<Double> deltaBias;
  double netInput;
  double activation;
  private List<Axon> inputAxons;
  private List<Axon> outputAxons;
  private ActivationFunction function;

  /**
   * Constructor
   * @param function Activation function
   */
  public Neuron(ActivationFunction function) {
    this.function = function;
    bias = Math.random();
    inputAxons = new ArrayList<>();
    outputAxons = new ArrayList<>();
    error = 0.0;
    deltaBias = IntStream.range(0, 2).mapToDouble(i -> 0.0)
        .boxed().collect(Collectors.toList());
    deltaWeight = IntStream.range(0, 2).mapToObj(i -> new ArrayList<Double>())
        .collect(Collectors.toList());
    netInput = 0.0;
    activation = 0.0;
  }

  /**
   * Add an input axon connection to the neuron
   * @param a Axon object
   */
  public void addInputAxon(Axon a) {
    inputAxons.add(a);
    deltaWeight.get(0).add(0.0);
    deltaWeight.get(1).add(0.0);
  }

  /**
   * Get a list of all input axons for the neuron
   * @return  List of input axons
   */
  public List<Axon> getInputAxons() {
    return inputAxons;
  }

  /**
   * Add an output axon connection to the neuron
   * @param a Axon object
   */
  public void addOutputAxon(Axon a) {
    outputAxons.add(a);
  }

  /**
   * Get list of output axons from the neuron
   * @return List of output axons
   */
  public List<Axon> getOutputAxons() {
    return outputAxons;
  }

  /**
   * Get list of weight update vectors
   * @return List of weight update vectors
   */
  public List<List<Double>> getDeltaWeight() {
    return deltaWeight;
  }

  /**
   * Set single weight update value in vector in list of weight update values
   * @param i   List index
   * @param j   Weight index
   * @param dw  Update value
   */
  public void setDeltaWeight(int i, int j, double dw) {
    deltaWeight.get(i).set(j, dw);
  }

  /**
   * Set single bias update value
   * @param i   Bias index
   * @param db  Update value
   */
  public void setDeltaBias(int i, double db) {
    deltaBias.set(i, db);
  }

  /**
   * Get list of bias update values
   * @return List of bias update values
   */
  public List<Double> getDeltaBias() {
    return deltaBias;
  }

  /**
   * Randomize input axon weights
   */
  public void randomizeInputWeights() {
    inputAxons.forEach(Axon::randomizeWeight);
  }

  /**
   * Randomize output axon weights
   */
  public void randomizeOutputWeights() {
    outputAxons.forEach(Axon::randomizeWeight);
  }

  /**
   * Randomize all axon weights
   */
  public void randomizeWeights() {
    randomizeInputWeights();
    randomizeOutputWeights();
  }

  /**
   * Feedforward algorithm
   * @return Activation output
   */
  public double forward() {
    netInput = inputAxons.stream()
        .mapToDouble(axon -> axon.getWeight() *
            axon.getDest().getActivation())
        .sum() / inputAxons.size();
    netInput += bias;
    activation = function.getActivation(netInput);
    return activation;
  }

  /**
   * Reset all parameter update values to zero
   */
  public void resetDeltas() {
    deltaWeight = deltaWeight.stream().map(doubles -> IntStream.range(0, doubles.size())
            .mapToDouble(j -> 0.0).boxed().collect(Collectors.toList()))
        .collect(Collectors.toList());
    deltaBias = deltaBias.stream().map(db -> Math.abs(db) * 0).collect(Collectors.toList());
  }

  /**
   * Clear all axon connections
   */
  public void disconnect() {
    inputAxons = new ArrayList<>();
    outputAxons = new ArrayList<>();
  }

  /**
   * Get activation function
   * @return Activation function
   */
  public ActivationFunction getFunction() {
    return function;
  }

  /**
   * Set loss value
   * @param e Loss value
   */
  public void setError(double e) {
    error = e;
  }

  /**
   * Get loss value
   * @return Loss value
   */
  public double getError() {
    return error;
  }

  /**
   * Set bias value
   * @param b Bias value
   */
  public void setBias(double b) {
    bias = b;
  }

  /**
   * Get bias value
   * @return Bias value
   */
  public double getBias() {
    return bias;
  }

  /**
   * Get last activation value
   * @return Last activation value
   */
  public double getActivation() {
    return activation;
  }

  /**
   * Set activation value
   * @param a Activation value
   */
  public void setActivation(double a) {
    activation = a;
  }

  /**
   * Get last net input
   * @return Last net input
   */
  public double getNetInput() {
    return netInput;
  }
}
