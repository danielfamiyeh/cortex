package neuron.activation;

public class ReluFunction implements ActivationFunction{
    @Override
    public double getActivation(double net) {
        return (net > 0) ? net : 0.0;
    }

    @Override
    public double getDerivative(double net){
        return (net > 0) ? 1 : 0.0;
    }
}
