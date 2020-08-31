package datatest;

import data.MinMax;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MinMaxTest {
    private static List<Double> vector1;
    private static List<Double> vector2;
    private static List<Double> vector3;
    private static List<List<Double>> vectorData;

    @BeforeEach
    public void setUp(){
        vector1 = new ArrayList<>(3);
        vector2 = new ArrayList<>(3);
        vector3 = new ArrayList<>(3);
        vectorData = new ArrayList<>();

        Collections.addAll(vector1, 3.0, 1.0, 7.0);
        Collections.addAll(vector2, 2.0, 9.0, 4.0);
        Collections.addAll(vector3, -6.0, -2.5, -3.0);
        Collections.addAll(vectorData, vector1, vector2, vector3);
    }

    @Test
    public void testNormalise(){
        List<Double> vec1Nomalised = new ArrayList<>(3);
        List<Double> vec2Nomalised = new ArrayList<>(3);
        List<Double> vec3Nomalised = new ArrayList<>(3);
        List<List<Double>> normalised = new ArrayList<>(3);

        Collections.addAll(vec1Nomalised, 1.0, (3.5/11.5), 1.0);
        Collections.addAll(vec2Nomalised, (8.0/9.0), 1.0, (7.0/10.0));
        Collections.addAll(vec3Nomalised, 0.0, 0.0, 0.0);
        Collections.addAll(normalised, vec1Nomalised, vec2Nomalised, vec3Nomalised);

        MinMax.scaleVectors(vectorData);

        for(int i=0; i<normalised.size(); i++){
            Assertions.assertEquals(normalised.get(i), vectorData.get(i));
        }
    }
}
