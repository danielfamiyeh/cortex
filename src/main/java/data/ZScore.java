package data;

import java.util.ArrayList;
import java.util.List;

public class ZScore {
  public static void standardiseVectors(List<List<Double>> vectors) {
    int numVectors = vectors.size();
    int length = vectors.get(0).size();

    List<Double> mean = new ArrayList<>(vectors.get(0));
    List<Double> stdDev = new ArrayList<>(length);

    // Sum components
    for (int vec = 1; vec < numVectors; vec++) {
      for (int comp = 0; comp < length; comp++) {
        mean.set(comp, mean.get(comp) + vectors.get(vec).get(comp));
      }
    }

    // Get mean and standard deviation
    for (int comp = 0; comp < length; comp++) {
      stdDev.add(0.0);
      mean.set(comp, mean.get(comp) / numVectors);

      for (List<Double> vector : vectors) {
        stdDev.set(comp, stdDev.get(comp) +
            Math.pow((vector.get(comp) - mean.get(comp)), 2)
                / numVectors);
      }
      stdDev.set(comp, Math.sqrt(stdDev.get(comp)));
    }

    stdDev.forEach(System.out::println);

    // Scale
    for (List<Double> vector : vectors) {
      for (int comp = 0; comp < length; comp++) {
        vector.set(comp, (vector.get(comp)
            - mean.get(comp)) / stdDev.get(comp));
      }
    }
  }

}
