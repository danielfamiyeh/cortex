package datatest;

import data.FileIO;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

public class FileIOTest {

  @Test
  public void testFileIORead() {
    double[][] xorFeatureVectors = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    List<List<Double>> xorData = FileIO.read("src/datasets/xor.csv");

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 2; j++) {
        Assertions.assertEquals(xorFeatureVectors[i][j],
            xorData.get(i).get(j));
      }
    }
  }
}
