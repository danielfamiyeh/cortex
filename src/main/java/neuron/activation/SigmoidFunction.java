package neuron.activation;

public class SigmoidFunction implements ActivationFunction {
  @Override
  public double getActivation(double net) {
    return 1 / (1 + Math.pow(Math.E, -net));
  }

  @Override
  public double getDerivative(double net) {
    return getActivation(net) * (1 - getActivation(net));
  }
}
