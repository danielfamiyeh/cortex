package neuron;
import neuron.activation.ActivationFunction;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Layer {
    private List<Double> activation;
    private List<Neuron> neuronList;

    public Layer(int size, ActivationFunction function){
        activation = IntStream.range(0, size)
                .mapToDouble(i -> 0.0)
                .boxed().collect(Collectors.toList());
        neuronList = IntStream.range(0, size)
                .mapToObj(i -> new Neuron(function))
                .collect(Collectors.toList());
    }

    public void forward(){
        for(int neuron=0; neuron<neuronList.size(); neuron++){
            activation.set(neuron, neuronList.get(neuron).forward());
        }
    }

    public void connect(
            Layer dest, int sourceIndex,
            int destIndex, double weight){
        try {
            Neuron sourceNeuron = neuronList.get(sourceIndex);
            Neuron destNeuron = dest.neuronList.get(destIndex);

            Axon sourceToDest = new Axon(weight, destNeuron);
            Axon destToSource = new Axon(weight, sourceNeuron);

            sourceNeuron.addOutputAxon(sourceToDest);
            destNeuron.addInputAxon(destToSource);

        } catch(ArrayIndexOutOfBoundsException aioobe){
            System.out.println("Invalid index passed to Layer.connect() method");
        }
    }

    public void connect(Layer dest, double weight){
        for(int sourceIndex=0;
            sourceIndex<neuronList.size();
            sourceIndex++){
            for(int destIndex=0;
                destIndex<dest.neuronList.size();
                destIndex++){
                connect(dest, sourceIndex, destIndex, weight);
            }
        }
    }

    public void connect(Layer dest){
        connect(dest, Math.random() - 0.5);
    }

    public List<Double> getActivation(){
        return activation;
    }

    public void setActivation(List<Double> a){
        for(int i=0; i<neuronList.size(); i++){
            neuronList.get(i).setActivation(a.get(i));
            activation.set(i, a.get(i));
        }
    }

    public int getSize(){
        return neuronList.size();
    }

    public void setBias(double b){
        neuronList.forEach(neuron -> neuron.setBias(b));
    }

    public List<Neuron> getNeuronList(){
        return neuronList;
    }
}
