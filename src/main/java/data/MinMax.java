package data;

import java.util.ArrayList;
import java.util.List;

public class MinMax {
    public static void scaleVectors(List<List<Double>> vectors){
        int numVectors = vectors.size();
        int length = vectors.get(0).size();

        List<Double> minVals = new ArrayList<>(length);
        List<Double> maxVals = new ArrayList<>(length);
        List<Double> ranges = new ArrayList<>(length);
        List<Double> current;

        // Get min-max values
        for(int i=0; i<numVectors; i++){
            current = vectors.get(i);
            for(int j=0; j<length; j++){
                double comp = current.get(j);
                if(i == 0){
                    minVals.add(comp);
                    maxVals.add(comp);
                } else{
                    double tentativeMin = minVals.get(j);
                    double tentativeMax = maxVals.get(j);
                    if(comp < tentativeMin){
                        minVals.set(j, comp);
                    }
                    if(comp > tentativeMax){
                        maxVals.set(j, comp);
                    }
                }
            }
        }

        // Calculate and store ranges
        for(int i=0; i<length; i++){
            ranges.add(maxVals.get(i) - minVals.get(i));
        }

        // Scale feature vectors according to ranges
        for (int i=0; i<numVectors; i++) {
            current = vectors.get(i);
            for (int j=0; j<length; j++) {
                double range = ranges.get(j);
                double comp = current.get(j);
                current.set(j, ((comp - minVals.get(j))/range));
            }
        }

    }
}
