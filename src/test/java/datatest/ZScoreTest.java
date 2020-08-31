package datatest;

import data.ZScore;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class ZScoreTest {
    private static List<Double> vector1;
    private static List<Double> vector2;
    private static List<Double> vector3;
    private static List<List<Double>> vectorData;

    private static double stdDev(double[] nums, double mean){
        return Math.sqrt(Arrays.stream(nums)
                .map(d -> Math.pow(d - mean, 2)/nums.length)
                .sum());
    }

    private static void initVectors(){
        vector1 = new ArrayList<>(3);
        vector2 = new ArrayList<>(3);
        vector3 = new ArrayList<>(3);
        vectorData = new ArrayList<>();

        Collections.addAll(vector1, 3.0, 1.0, 3.0);
        Collections.addAll(vector2, 2.0, 7.0, 6.0);
        Collections.addAll(vector3, 4.0, 0.0, 7.0);
        Collections.addAll(vectorData, vector1, vector2, vector3);
    }

    @BeforeAll
    public static void setUp(){
        initVectors();
    }

    @Test
    public void testVectorStandardise(){
        List<List<Double>> standardised = new ArrayList<>(3);
        double std1 = stdDev(new double[]{3.0, 2.0, 4.0}, 3);
        double std2 = stdDev(new double[]{1.0, 7.0, 0.0}, (8.0/3.0));
        double std3 = stdDev(new double[]{3.0, 6.0, 7.0}, (16.0/3.0));

        List<Double> v1 = Arrays.asList(0.0,
                (1.0-(8.0/3.0))/(std2),
                (3.0-(16.0/3.0))/Math.sqrt(2.8888888888888889));

        List<Double> v2 = Arrays.asList((-1.0/std1),
                (7.0-(8.0/3.0))/(std2),
                (6.0-(16.0/3.0))/Math.sqrt(2.8888888888888889));

        List<Double> v3 = Arrays.asList((1.0/std1),
                (0.0-(8.0/3.0))/(std2),
                (7.0-(16.0/3.0))/Math.sqrt(2.8888888888888889));

        Collections.addAll(standardised, v1, v2, v3);
        ZScore.standardiseVectors(vectorData);

        vectorData.forEach(System.out::println);


        for(int i=0; i<vectorData.size(); i++){
            Assertions.assertEquals(standardised.get(i), vectorData.get(i));
        }
    }
}
