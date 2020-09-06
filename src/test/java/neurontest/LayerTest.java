package neurontest;

import neuron.Layer;
import neuron.Neuron;
import neuron.activation.ReluFunction;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

public class LayerTest {

    private static Layer inputLayer;
    private static Layer hiddenLayer;
    private static Layer outputLayer;

    @BeforeEach
    public void setUp(){
        inputLayer = new Layer(2, null);
        hiddenLayer = new Layer(2, new ReluFunction());
        outputLayer = new Layer(1, new ReluFunction());
    }

    @Test
    public void testInitLayer(){
        Assertions.assertEquals(2, inputLayer.getSize());
        Assertions.assertEquals(2, hiddenLayer.getSize());
        Assertions.assertEquals(1, outputLayer.getSize());
    }

    @Test
    public void testGetActivation(){
        Assertions.assertEquals(Arrays.asList(0.0, 0.0),
                inputLayer.getActivation());
    }

    @Test
    public void testSetActivation(){
        List<Double> newInput = Arrays.asList(1.3, 2.2);
        List<Double> newHidden = Arrays.asList(-0.7, 3.5);

        inputLayer.setActivation(newInput);
        hiddenLayer.setActivation(newHidden);

        Assertions.assertEquals(newInput, inputLayer.getActivation());
        Assertions.assertEquals(newHidden, hiddenLayer.getActivation());
    }

    @Test
    public void testSetBias(){
        inputLayer.setBias(0.3);
        hiddenLayer.setBias(-0.3);

        inputLayer.getNeuronList().forEach(neuron ->
                Assertions.assertEquals(0.3, neuron.getBias()));

        hiddenLayer.getNeuronList().forEach(neuron ->
                Assertions.assertEquals(-0.3, neuron.getBias()));
    }

    @Test
    public void testConnect(){
        inputLayer.connect(hiddenLayer, 0, 1, 2.3);
        hiddenLayer.connect(outputLayer, -1.7);

        Assertions.assertEquals(inputLayer.getNeuronList().get(0).getOutputAxons()
        .get(0).getDest(), hiddenLayer.getNeuronList().get(1));

        Assertions.assertEquals(hiddenLayer.getNeuronList().get(1).getInputAxons()
        .get(0).getDest(), inputLayer.getNeuronList().get(0));

        Assertions.assertEquals(inputLayer.getNeuronList().get(0).getOutputAxons()
                .get(0).getWeight(), 2.3);

        Assertions.assertEquals(hiddenLayer.getNeuronList().get(1).getInputAxons()
                .get(0).getWeight(), 2.3);

        Assertions.assertEquals(hiddenLayer.getNeuronList().get(0).getOutputAxons()
                .get(0).getDest(), outputLayer.getNeuronList().get(0));

        Assertions.assertEquals(hiddenLayer.getNeuronList().get(1).getOutputAxons()
                .get(0).getDest(), outputLayer.getNeuronList().get(0));

        Assertions.assertEquals(hiddenLayer.getNeuronList().get(0).getOutputAxons()
                .get(0).getWeight(), -1.7);

        Assertions.assertEquals(hiddenLayer.getNeuronList().get(1).getOutputAxons()
                .get(0).getWeight(), -1.7);

    }

    @Test
    public void testForward(){
        inputLayer.setBias(1);
        inputLayer.connect(hiddenLayer, 1.0);
        hiddenLayer.connect(outputLayer);

        inputLayer.setActivation(Arrays.asList(0.3, 0.72));
        hiddenLayer.forward();
        outputLayer.forward();
        System.out.println(outputLayer.getActivation());
    }

    @Test
    public void testDisconnect(){
        inputLayer.connect(hiddenLayer);
        hiddenLayer.connect(outputLayer);
        hiddenLayer.disconnect();

        List<Neuron> hiddenLayerNeurons = hiddenLayer.getNeuronList();
        hiddenLayerNeurons.forEach(n ->
            Assertions.assertEquals(0, n.getOutputAxons().size()));
    }
}
