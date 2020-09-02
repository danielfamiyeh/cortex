package optimiser.algorithm;

import neuron.Axon;
import neuron.Neuron;

import java.util.List;
import java.util.function.BiConsumer;

public class OptimAlgo {
    private static double beta = 0.5;
    private static double epsilon = 1E-7;

    public static BiConsumer<Neuron, Double> rms = (neuron, alpha) -> {
        double oldBias = beta * neuron.getDeltaBias();
        double newBias = (1 - beta) * Math.pow(neuron.getError(), 2);
        List<Axon> inputAxons = neuron.getInputAxons();

        for(int i=0; i<inputAxons.size(); i++){
            Axon axon = inputAxons.get(i);
            double delW = axon.getDest().getActivation() * neuron.getError();
            double oldDw = beta * neuron.getDeltaWeight().get(i);
            double newDw = (1-beta) *
                    Math.pow(delW, 2);
            neuron.setDeltaWeight(i, oldDw + newDw);
            axon.decrementWeight(alpha * (
                    delW/(Math.sqrt(neuron.getDeltaWeight().get(i)) + epsilon)
                    ));
        }

        neuron.setDeltaBias(oldBias + newBias);
        neuron.setBias(neuron.getBias() - alpha * (
              neuron.getError()/
                      (Math.sqrt(neuron.getDeltaBias()) + epsilon)
              ));
    };

    public static BiConsumer<Neuron, Double> momentum = (neuron, alpha) -> {
        double oldBias = beta * neuron.getDeltaBias();
        double newBias = (1 - beta) * neuron.getError();

        List<Axon> inputAxons = neuron.getInputAxons();
        for(int i=0; i<inputAxons.size(); i++){
            Axon axon = inputAxons.get(i);
            double oldDw = beta * neuron.getDeltaWeight().get(i);
            double newDw = (1-beta) * axon.getDest().getActivation() * neuron.getError();
            neuron.setDeltaWeight(i, oldDw + newDw);
            axon.decrementWeight(alpha * neuron.getDeltaWeight().get(i));
        }
        neuron.setDeltaBias(oldBias + newBias);
        neuron.setBias(neuron.getBias() - (alpha * neuron.getDeltaBias()));
    };

    public static BiConsumer<Neuron, Double> sgd = (neuron, alpha) -> {
        double error = neuron.getError();
        List<Axon> inputAxons = neuron.getInputAxons();
        for(Axon axon : inputAxons){
            axon.decrementWeight(axon.getDest().getActivation() * alpha * error);
        }
        neuron.setBias(neuron.getBias() - alpha * error);
    };
}
