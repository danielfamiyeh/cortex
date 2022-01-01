package datatest;

import data.MinMax;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class MinMaxTest {
  private static List<Double> vector1;
  private static List<Double> vector2;
  private static List<Double> vector3;
  private static List<List<Double>> vectorData;

  private static List<List<Double>> matrix1;
  private static List<List<Double>> matrix2;
  private static List<List<Double>> matrix3;
  private static List<List<List<Double>>> matrixData;

  private static void initVectors() {
    vector1 = new ArrayList<>(3);
    vector2 = new ArrayList<>(3);
    vector3 = new ArrayList<>(3);
    vectorData = new ArrayList<>();

    Collections.addAll(vector1, 3.0, 1.0, 7.0);
    Collections.addAll(vector2, 2.0, 9.0, 4.0);
    Collections.addAll(vector3, -6.0, -2.5, -3.0);
    Collections.addAll(vectorData, vector1, vector2, vector3);
  }

  private static void initMatrices() {
    matrix1 = new ArrayList<>();
    matrix2 = new ArrayList<>();
    matrix3 = new ArrayList<>();
    matrixData = new ArrayList<>();

    matrix1.add(Arrays.asList(3.0, 1.0, 3.0));
    matrix1.add(Arrays.asList(2.0, 7.0, 6.0));
    matrix1.add(Arrays.asList(4.0, 6.0, 7.0));

    matrix2.add(Arrays.asList(2.0, 4.0, 3.0));
    matrix2.add(Arrays.asList(3.0, 3.0, 1.0));
    matrix2.add(Arrays.asList(1.0, 9.0, 12.0));

    matrix3.add(Arrays.asList(-14.0, 0.0, 0.0));
    matrix3.add(Arrays.asList(1.0, 15.0, 6.0));
    matrix3.add(Arrays.asList(-2.3, 3.0, 3.7));

    Collections.addAll(matrixData, matrix1, matrix2, matrix3);
  }

  @BeforeAll
  public static void setUp() {
    initVectors();
    initMatrices();
  }


  @Test
  public void testScaleVectors() {
    List<Double> vec1Nomalised = new ArrayList<>(3);
    List<Double> vec2Nomalised = new ArrayList<>(3);
    List<Double> vec3Nomalised = new ArrayList<>(3);
    List<List<Double>> normalised = new ArrayList<>(3);

    Collections.addAll(vec1Nomalised, 1.0, (3.5 / 11.5), 1.0);
    Collections.addAll(vec2Nomalised, (8.0 / 9.0), 1.0, (7.0 / 10.0));
    Collections.addAll(vec3Nomalised, 0.0, 0.0, 0.0);
    Collections.addAll(normalised, vec1Nomalised, vec2Nomalised, vec3Nomalised);

    MinMax.scaleVectors(vectorData);

    for (int i = 0; i < normalised.size(); i++) {
      Assertions.assertEquals(normalised.get(i), vectorData.get(i));
    }
  }

  @Test
  public void testScaleMatrices() {
    List<List<Double>> mat1Normalised = new ArrayList<>();
    List<List<Double>> mat2Normalised = new ArrayList<>();
    List<List<Double>> mat3Normalised = new ArrayList<>();
    List<List<List<Double>>> normalisedMatrices = new ArrayList<>();

    mat1Normalised.add(Arrays.asList(1.0, 0.25, 1.0));
    mat1Normalised.add(Arrays.asList(0.5, (1.0 / 3.0), 1.0));
    mat1Normalised.add(Arrays.asList(1.0, 0.5, (3.3 / 8.3)));

    mat2Normalised.add(Arrays.asList((16.0 / 17.0), 1.0, 1.0));
    mat2Normalised.add(Arrays.asList(1.0, 0.0, 0.0));
    mat2Normalised.add(Arrays.asList((11.0 / 21.0), 1.0, 1.0));

    mat3Normalised.add(Arrays.asList(0.0, 0.0, 0.0));
    mat3Normalised.add(Arrays.asList(0.0, 1.0, 1.0));
    mat3Normalised.add(Arrays.asList(0.0, 0.0, 0.0));

    Collections.addAll(normalisedMatrices, mat1Normalised,
        mat2Normalised, mat3Normalised);

    MinMax.scaleMatrices(matrixData);

    for (int i = 0; i < matrixData.size(); i++) {
      for (int j = 0; j < matrixData.get(i).size(); j++) {
        Assertions.assertEquals(normalisedMatrices.get(i).get(j),
            matrixData.get(i).get(j));
      }
    }

  }
}
