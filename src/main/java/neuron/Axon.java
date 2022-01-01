package neuron;

/**
 * Class representing a connection between two Neurons
 */
public class Axon {
  private double weight;
  private Neuron dest;

  /**
   *
   * @param weight Weight of connection
   * @param dest   Destination neuron
   */
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

  public void randomizeWeight() {
    weight = Math.random() - 0.5;
  }
}
