package optimiser.algorithm;

import neuron.Axon;
import neuron.Neuron;

import java.util.List;
import java.util.function.BiConsumer;

public class OptimAlgo {
    private static double beta = 0.5;
    public static BiConsumer<Neuron, Double> momentum = (neuron, alpha) -> {
        double oldBias = beta * neuron.getDeltaBias();
        double newBias = (1 - beta) * neuron.getError();

        List<Axon> inputAxons = neuron.getInputAxons();
        for(int i=0; i<inputAxons.size(); i++){
            Axon axon = inputAxons.get(i);
            double oldDw = beta * neuron.getDeltaWeight().get(i);
            double newDw = (1-beta) * axon.getDest().getActivation() * neuron.getError();
            neuron.setDeltaWeight(i, alpha *(oldDw + newDw));
            axon.decrementWeight(neuron.getDeltaWeight().get(i));
        }
        neuron.setDeltaBias(alpha * (oldBias + newBias));
        neuron.setBias(neuron.getBias() - neuron.getDeltaBias());
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
