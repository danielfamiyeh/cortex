package optimisertest;

import neuron.Layer;
import neuron.activation.ReluFunction;
import neuron.activation.SigmoidFunction;
import optimiser.algorithm.SGD;
import optimiser.loss.MSEFunction;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class OptimiserTest {
    private static List<List<Double>> xorDataset;
    private static List<List<Double>> labels;
    private static List<Layer> network;
    private static SGD optimiser;

    @BeforeAll
    public static void setUp(){
        xorDataset = new ArrayList<>();
        labels = new ArrayList<>();
        network = new ArrayList<>();

        optimiser = new SGD();

        Collections.addAll(xorDataset, Arrays.asList(0.0, 0.0),
                Arrays.asList(0.0, 1.0),
                Arrays.asList(1.0, 0.0),
                Arrays.asList(1.0, 1.0));

        Collections.addAll(labels, Arrays.asList(0.0),
                Arrays.asList(1.0),
                Arrays.asList(1.0),
                Arrays.asList(0.0));

        Collections.addAll(network, new Layer(2, null),
                new Layer(2, new ReluFunction()),
                new Layer(1, new ReluFunction()));

        for(int i=0; i<network.size()-1; i++){
            network.get(i).connect(network.get(i+1));
        }


    }

    @Test
    public void testSGD(){
        optimiser.optimiseDNN(network, xorDataset,
                labels, new MSEFunction(), 5000);
    }
}
