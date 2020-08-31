package neuron;

import neuron.activation.ActivationFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class Neuron {
    double bias;
    double netInput;
    double activation;
    private List<Axon> inputAxons;
    private List<Axon> outputAxons;
    private ActivationFunction function;

    public Neuron(ActivationFunction function){
        this.function = function;
        bias = Math.random() - 0.5;
        inputAxons = new ArrayList<>();
        outputAxons = new ArrayList<>();
        netInput = 0.0;
        activation = 0.0;
    }

    public void addInputAxon(Axon a){
        inputAxons.add(a);
    }

    public List<Axon> getInputAxons(){
        return inputAxons;
    }

    public void addOutputAxon(Axon a){
        outputAxons.add(a);
    }

    public List<Axon> getOutputAxons(){
        return outputAxons;
    }

    public void randomiseInputWeights(){
        inputAxons.forEach(Axon::randomiseWeight);
    }

    public void randomiseOutputWeights(){
        outputAxons.forEach(Axon::randomiseWeight);
    }

    public void randomiseWeights(){
        randomiseInputWeights();
        randomiseOutputWeights();
    }

    public void forward(){
        netInput = (inputAxons.stream()
                .mapToDouble(axon -> axon.getWeight() *
                        axon.getDest().getActivation())
                .sum() + bias)/ inputAxons.size();
        activation = function.getActivation(netInput);
    }

    public void setBias(double b){
        bias = b;
    }

    public double getBias(){
        return bias;
    }

    public double getActivation(){
        return activation;
    }

    public double getNetInput(){
        return netInput;
    }

    public void setActivation(double a){
        activation = a;
    }
}
