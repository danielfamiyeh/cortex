package neuron;

import neuron.activation.ActivationFunction;
import neuron.activation.ReluFunction;

import java.util.ArrayList;
import java.util.List;

public class Neuron {
    double bias;
    double error;
    List<Double> deltaWeight;
    double deltaBias;
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
        deltaBias = 0.0;
        deltaWeight = new ArrayList<>();
        netInput = 0.0;
        activation = 0.0;
    }

    public void addInputAxon(Axon a){
        inputAxons.add(a);
        deltaWeight.add(0.0);
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

    public List<Double> getDeltaWeight(){
        return deltaWeight;
    }

    public void setDeltaWeight(int index, double dw){
        deltaWeight.set(index, dw);
    }

    public void setDeltaBias(double db){
        deltaBias = db;
    }

    public double getDeltaBias(){
        return deltaBias;
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
