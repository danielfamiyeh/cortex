package neuron.activation;

public interface ActivationFunction {
  double getActivation(double net);

  double getDerivative(double net);

}
