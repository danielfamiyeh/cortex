package neuron;

import neuron.activation.ActivationFunction;
import neuron.activation.ReluFunction;

import java.util.ArrayList;
import java.util.List;

public class Neuron {
    double bias;
    double error;
    double netInput;
    double activation;
    private List<Axon> inputAxons;
    private List<Axon> outputAxons;
    private ActivationFunction function;

    public Neuron(ActivationFunction function){
        this.function = function;
        bias = Math.random();
        inputAxons = new ArrayList<>();
        outputAxons = new ArrayList<>();
        error = 0.0;
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

    public double forward(){
        netInput = inputAxons.stream()
                .mapToDouble(axon -> axon.getWeight() *
                        axon.getDest().getActivation())
                .sum() / inputAxons.size();
        netInput += bias;
        activation = function.getActivation(netInput);
        return activation;
    }

    public void updateWeights(double alpha){
        for(Axon axon : inputAxons){
            axon.decrementWeight(axon.getDest().getActivation() * alpha * error);
        }
        bias -= alpha * error;
    }

    public ActivationFunction getFunction(){
        return function;
    }

    public void setError(double e){
        error = e;
    }

    public double getError(){
        return error;
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
