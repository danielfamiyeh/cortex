package optimisertest;

import neuron.Layer;
import neuron.activation.ReluFunction;
import optimiser.DNNOptimiser;
import optimiser.algorithm.OptimAlgo;
import optimiser.loss.MSEFunction;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class OptimiserTest {
    private static List<List<Double>> xorDataset;
    private static List<List<Double>> labels;
    private static List<Layer> network;
    private static DNNOptimiser underTest;

    @BeforeEach
    public void setUp(){
        xorDataset = new ArrayList<>();
        labels = new ArrayList<>();
        network = new ArrayList<>();

        underTest = new DNNOptimiser();

        Collections.addAll(xorDataset, Arrays.asList(0.0, 0.0),
                Arrays.asList(0.0, 1.0),
                Arrays.asList(1.0, 0.0),
                Arrays.asList(1.0, 1.0));

        Collections.addAll(labels, Arrays.asList(0.0),
                Arrays.asList(1.0),
                Arrays.asList(1.0),
                Arrays.asList(0.0));

        Collections.addAll(network, new Layer(2, null),
                new Layer(3, new ReluFunction()),
                new Layer(1, new ReluFunction()));

        for(int i=0; i<network.size()-1; i++){
            network.get(i).connect(network.get(i+1));
        }


    }

    @Test
    public void testSGD(){
       underTest.optimise(
               network, xorDataset,
               labels, new MSEFunction(), 100000, 0.01,
               OptimAlgo.sgd
       );
    }

    @Test
    public void testSGDMomentum(){
        System.out.println("\n");
        underTest.optimise(
                network, xorDataset,
                labels, new MSEFunction(), 100000, 0.01,
                OptimAlgo.momentum
        );
    }

    @Test
    public void testRMSProp(){
        System.out.println("\n");
        underTest.optimise(
                network, xorDataset,
                labels, new MSEFunction(), 10000, 0.01,
                OptimAlgo.rms
        );
    }

    @Test
    public void testAdam(){
        System.out.println("\n");
        underTest.optimise(
                network, xorDataset,
                labels, new MSEFunction(), 10000, 0.01,
                OptimAlgo.adam
        );
    }
}
