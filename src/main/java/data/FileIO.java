package data;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.stream.Collectors;

public class FileIO{
    public static List<List<Double>> read(String path){
        List<List<Double>> data = new ArrayList<>();
        try {
            Scanner reader = new Scanner(new File(path));
            while(reader.hasNext()){
                data.add(Arrays.stream(reader.next().split(","))
                        .mapToDouble(Double::parseDouble)
                        .boxed().collect(Collectors.toList()));
            }

            reader.close();

        } catch (FileNotFoundException ffne){
            System.out.println("File not found exception in FileIO,read().");
        }

        return data;
    }
}