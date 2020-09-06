package neuron;

import neuron.activation.ActivationFunction;
import neuron.activation.ReluFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Neuron {
    double bias;
    double error;
    List<List<Double>> deltaWeight;
    List<Double> deltaBias;
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
        deltaBias = IntStream.range(0, 2).mapToDouble(i -> 0.0)
                .boxed().collect(Collectors.toList());
        deltaWeight = IntStream.range(0, 2).mapToObj(i -> new ArrayList<Double>())
                .collect(Collectors.toList());
        netInput = 0.0;
        activation = 0.0;
    }

    public void addInputAxon(Axon a){
        inputAxons.add(a);
        deltaWeight.get(0).add(0.0);
        deltaWeight.get(1).add(0.0);
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

    public List<List<Double>> getDeltaWeight(){
        return deltaWeight;
    }

    public void setDeltaWeight(int i, int j, double dw){
        deltaWeight.get(i).set(j, dw);
    }

    public void setDeltaBias(int i, double db){
        deltaBias.set(i, db);
    }

    public List<Double> getDeltaBias(){
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

    public void resetDeltas(){
        deltaWeight = deltaWeight.stream().map(doubles -> IntStream.range(0, doubles.size())
                .mapToDouble(j -> 0.0).boxed().collect(Collectors.toList()))
                .collect(Collectors.toList());
        deltaBias = deltaBias.stream().map(db -> Math.abs(db) * 0).collect(Collectors.toList());
    }

    public void disconnect(){
        inputAxons = new ArrayList<>();
        outputAxons = new ArrayList<>();
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
