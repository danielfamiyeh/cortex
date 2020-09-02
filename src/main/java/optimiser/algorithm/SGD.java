package optimiser.algorithm;

import neuron.Axon;
import neuron.Layer;
import neuron.Neuron;
import optimiser.loss.LossFunction;

import java.util.List;
import java.util.function.BiConsumer;

public class SGD implements Optimiser {
    private Double beta = null;

    @Override
    public void optimiseDNN(
            List<Layer> network, List<List<Double>> dataset, List<List<Double>> labels,
            LossFunction lossFunction, int numEpochs, double alpha) {

        int numExamples = dataset.size();
        Layer inputLayer = network.get(0);
        Layer outputLayer = network.get(network.size() - 1);
        List<Double> errorVector;
        List<Double> yHat;

        for(int epoch=0; epoch<numEpochs; epoch++){
            double cost = 0.0;

            for(int i=0; i<numExamples; i++){
                List<Double> feature = dataset.get(i);
                inputLayer.setActivation(feature);
                for(Layer layer : network){
                    layer.forward();
                }
                yHat = outputLayer.getActivation();
                cost += lossFunction.getLoss(yHat,
                        labels.get(i));

                errorVector = lossFunction.getDerivative(yHat, labels.get(i));

                for(int j=0; j<errorVector.size(); j++){
                    Neuron outputNeuron = outputLayer.getNeuronList().get(j);
                    errorVector.set(j, errorVector.get(j) *
                            outputNeuron.getFunction().getDerivative(
                                    outputNeuron.getNetInput()));
                }
                if(epoch == numEpochs-1) {
                    System.out.println(yHat);
                    System.out.println("Error vector" + errorVector);
                }
                // Backprop
                outputLayer.setError(errorVector);
                for(int j=network.size()-2; j>0; j--){
                    Layer hiddenLayer = network.get(j);
                    for(Neuron neuron : hiddenLayer.getNeuronList()){
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
                for(int j=network.size()-1; j>0; j--){
                    network.get(j).updateWeights((beta == null) ?
                            sgdUpdate : sgdMomentum, alpha);
                }
            }
            cost /= numExamples;
    //        System.out.println("Epoch: " + (epoch+1));
      //      System.out.println("Average Error: " + cost + "\n");
        }
    }

    public void setBeta(double b){
        beta = b;
    }

    private BiConsumer<Neuron, Double> sgdMomentum = (neuron, alpha) -> {
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

    private BiConsumer<Neuron, Double> sgdUpdate = (neuron, alpha) -> {
        double error = neuron.getError();
        List<Axon> inputAxons = neuron.getInputAxons();
        for(Axon axon : inputAxons){
            axon.decrementWeight(axon.getDest().getActivation() * alpha * error);
        }
        neuron.setBias(neuron.getBias() - alpha * error);
    };
}
