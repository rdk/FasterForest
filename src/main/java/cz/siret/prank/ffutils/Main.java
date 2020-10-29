package cz.siret.prank.ffutils;

import cz.siret.prank.fforest2.FasterForest2;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.InputStream;

import static java.lang.System.out;

/**
 *
 */
public class Main {

    static Instances loadData(String fileName) {
        try (InputStream is = new BufferedInputStream(new FileInputStream(fileName))) {

            ConverterUtils.DataSource source = new ConverterUtils.DataSource(is);
            Instances data = source.getDataSet();

            // setting class attribute if the data format does not provide this information
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);

            return data;
        } catch (Exception e) {
            throw new RuntimeException("Failed to load dataset.", e);
        } 
    }


    public static void main(String[] args) {

        out.println("FasterForest library benchmark.");
        out.println("usage: \n    java -jar FasterForest.xx.jar <n_threads> <n_trees> <arff_dataset>");

        int threads = Integer.parseInt(args[0]);
        int trees = Integer.parseInt(args[1]);
        String fdata = args[2];

        out.printf("Training FasterForest2 with %d threads and %d trees on dataset %s%n\n", threads, trees, fdata);

        FasterForest2 forest = new FasterForest2();
        forest.setNumThreads(threads);
        forest.setNumTrees(trees);

        Instances data = loadData(fdata);

        try {
            ATimer timer = ATimer.startTimer();
            forest.buildClassifier(data);
            out.println("Forest trained in " + timer.getFormatted());
        } catch (Exception e) {
            throw new RuntimeException("Failed to train forest.", e);
        }


    }
    
}
