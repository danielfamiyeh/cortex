package neuron;

public class Axon {
  private double weight;
  private Neuron dest;

  public Axon(double weight, Neuron dest) {
    this.weight = weight;
    this.dest = dest;
  }

  public Neuron getDest() {
    return dest;
  }

  public double getWeight() {
    return weight;
  }

  public void setWeight(double weight) {
    this.weight = weight;
  }

  public void decrementWeight(double d) {
    weight -= d;
  }

  public void randomiseWeight() {
    weight = Math.random() - 0.5;
  }
}
