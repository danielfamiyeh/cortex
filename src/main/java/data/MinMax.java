package data;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
        for (List<Double> vector : vectors) {
            current = vector;
            for (int i = 0; i < length; i++) {
                double range = ranges.get(i);
                double comp = current.get(i);
                current.set(i, ((comp - minVals.get(i)) / range));
            }
        }

    }
    public static void scaleMatrices(List<List<List<Double>>> matrices){
        int numMatrices = matrices.size();
        int numRows = matrices.get(0).size();
        int numCols = matrices.get(0).get(0).size();
        List<List<Double>> minMatrix = new ArrayList<>();
        List<List<Double>> maxMatrix = new ArrayList<>();
        List<List<Double>> ranges = new ArrayList<>();
        List<List<Double>> currentMatrix;
        List<Double> currentRow;

        // Get min-max values
        for(int i=0; i<numMatrices; i++){
            currentMatrix = matrices.get(i);
            for(int row=0; row<numRows; row++){
                currentRow = currentMatrix.get(row);
                if(i > 0){
                    for(int col=0; col<numCols; col++) {
                        double currentMin = minMatrix.get(row).get(col);
                        double currentMax = maxMatrix.get(row).get(col);
                        double tentative = currentRow.get(col);

                        if(tentative > currentMax){
                            maxMatrix.get(row).set(col, tentative);
                        }
                        if(tentative < currentMin){
                            minMatrix.get(row).set(col, tentative);
                        }
                    }
                } else{
                    minMatrix.add(new ArrayList<>(currentRow));
                    maxMatrix.add(new ArrayList<>(currentRow));
                }
            }
        }

        // Get ranges
        for(int row=0; row<numRows; row++){
            int finalRow = row;
            ranges.add(IntStream.range(0, numCols)
            .mapToObj(i ->
                    maxMatrix.get(finalRow).get(i)
                    - minMatrix.get(finalRow).get(i))
            .collect(Collectors.toList()));
        }

        // Scale matrices
        for(int i=0; i<numMatrices; i++){
            int finalI = i;
            matrices.set(i, IntStream.range(0, numRows).mapToObj(
                    row -> IntStream.range(0, numCols).mapToDouble(
                            col -> (matrices.get(finalI).get(row).get(col) -
                                    minMatrix.get(row).get(col)) / ranges.get(
                                            row).get(col)
                            ).boxed().collect(Collectors.toList())
                    ).collect(Collectors.toList())
            );
        }
    }
}
