package optimiser.algorithm;

import neuron.Axon;
import neuron.Neuron;

import java.util.List;
import java.util.function.BiConsumer;

public class OptimAlgo {
    private static double beta = 0.5;
    private static double beta2 = 0.5;
    private static double epsilon = 1E-7;

    public static BiConsumer<Neuron, Double> adam = (neuron, alpha) -> {
        // Momentum bias update value
        double vDeltaBias = beta * neuron.getDeltaBias().get(0) +
                (1 - beta) * neuron.getError();
        // RMSProp bias update value
        double sDeltaBias = beta2 * neuron.getDeltaBias().get(1) +
                (1 - beta) * Math.pow(neuron.getError(), 2);

        List<Axon> inputAxons = neuron.getInputAxons();

        for(int i=0; i<inputAxons.size(); i++){
            Axon axon = inputAxons.get(i);
            double delW = axon.getDest().getActivation() * neuron.getError();
            double vDeltaWeight = (beta * neuron.getDeltaWeight().get(0).get(i)) +
                    (1 - beta) * (delW);
            double sDeltaWeight = (beta * neuron.getDeltaWeight().get(1).get(i)) +
                    (1 - beta) * Math.pow(delW, 2);

            neuron.setDeltaWeight(0, i, vDeltaWeight);
            neuron.setDeltaWeight(1, i, sDeltaWeight);

            axon.decrementWeight(alpha * (
                    vDeltaWeight/(Math.sqrt(sDeltaWeight) + epsilon)
                    ));
        }

        neuron.setDeltaBias(0, vDeltaBias);
        neuron.setDeltaBias(1, sDeltaBias);
        neuron.setBias(neuron.getBias() - alpha * (
                vDeltaBias / (Math.sqrt(sDeltaBias) + epsilon)
                ));
    };

    public static BiConsumer<Neuron, Double> rms = (neuron, alpha) -> {
        double oldBias = beta * neuron.getDeltaBias().get(0);
        double newBias = (1 - beta) * Math.pow(neuron.getError(), 2);
        List<Axon> inputAxons = neuron.getInputAxons();

        for(int i=0; i<inputAxons.size(); i++){
            Axon axon = inputAxons.get(i);
            double delW = axon.getDest().getActivation() * neuron.getError();
            double oldDw = beta * neuron.getDeltaWeight().get(0).get(i);
            double newDw = (1-beta) *
                    Math.pow(delW, 2);
            neuron.setDeltaWeight(0, i, oldDw + newDw);
            axon.decrementWeight(alpha * (
                    delW/(Math.sqrt(neuron.getDeltaWeight().get(0).get(i)) + epsilon)
                    ));
        }

        neuron.setDeltaBias(0,oldBias + newBias);
        neuron.setBias(neuron.getBias() - alpha * (
              neuron.getError()/
                      (Math.sqrt(neuron.getDeltaBias().get(0)) + epsilon)
              ));
    };

    public static BiConsumer<Neuron, Double> momentum = (neuron, alpha) -> {
        double oldBias = beta * neuron.getDeltaBias().get(0);
        double newBias = (1 - beta) * neuron.getError();

        List<Axon> inputAxons = neuron.getInputAxons();
        for(int i=0; i<inputAxons.size(); i++){
            Axon axon = inputAxons.get(i);
            double oldDw = beta * neuron.getDeltaWeight().get(0).get(i);
            double newDw = (1-beta) * axon.getDest().getActivation() * neuron.getError();
            neuron.setDeltaWeight(0, i, oldDw + newDw);
            axon.decrementWeight(alpha * neuron.getDeltaWeight().get(0).get(i));
        }
        neuron.setDeltaBias(0, oldBias + newBias);
        neuron.setBias(neuron.getBias() - (alpha * neuron.getDeltaBias().get(0)));
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
