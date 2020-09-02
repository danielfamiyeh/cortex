package optimiser;

import neuron.Layer;
import neuron.Neuron;
import optimiser.loss.LossFunction;

import java.util.List;
import java.util.function.BiConsumer;

public class DNNOptimiser {
    public DNNOptimiser(){}

    public void optimise(List<Layer> network, List<List<Double>> dataset,
                         List<List<Double>> labels, LossFunction lossFunction,
                         int numEpochs, double alpha, BiConsumer<Neuron, Double> algorithm){

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
                    network.get(j).updateWeights(algorithm, alpha);
                }
            }
            cost /= numExamples;
            //        System.out.println("Epoch: " + (epoch+1));
            //      System.out.println("Average Error: " + cost + "\n");
        }
    }
}
