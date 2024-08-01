package Java;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

public class Main{

    public static void main(String[] args){
        if(args.length != 1){
            System.out.println("Usage: java Main <name_of_file>");
            return;
        }

        String fileName = args[0];
        try{
            List<String> lines = Files.readAllLines(Paths.get(fileName));
            int size = Integer.parseInt(lines.get(0).trim());
            double[][] matrix1 = parseMatrix(lines, 1, size);
            double[][] matrix2 = parseMatrix(lines, 1 + size + 1, size); 

            ForkJoinPool pool = new ForkJoinPool();
            double[][] result = pool.invoke(new MatrixMultiplicationTask(matrix1, matrix2, 0, size));

            System.out.println(size);
            for (int i = 0; i < result.length; i++) {
                for (int j = 0; j < result[0].length; j++) {
                    System.out.print(result[i][j] + " ");
                }
                System.out.println();
            }
            pool.close();
        }catch (IOException e){
            e.printStackTrace();
        }
    }

    private static double[][] parseMatrix(List<String> lines, int startLine, int size){
        double[][] matrix = new double[size][size];

        for(int i = 0; i < size; i++){
            String line = lines.get(startLine + i);
            if(line.trim().isEmpty()){
                continue; // Skip empty lines
            }
            String[] elements = line.split(" ");
            if(elements.length != size){
                throw new IllegalArgumentException("Invalid matrix size");
            }
            for (int j = 0; j < size; j++){
                matrix[i][j] = Double.parseDouble(elements[j]);
            }
        }

        return matrix;
    }

    static class MatrixMultiplicationTask extends RecursiveTask<double[][]>{
        private static final int THRESHOLD = 100;
        private double[][] firstMatrix;
        private double[][] secondMatrix;
        private int startRow;
        private int endRow;

        public MatrixMultiplicationTask(double[][] firstMatrix, double[][] secondMatrix, int startRow, int endRow){
            this.firstMatrix = firstMatrix;
            this.secondMatrix = secondMatrix;
            this.startRow = startRow;
            this.endRow = endRow;
        }

        @Override
        protected double[][] compute(){
            int size = firstMatrix.length;
            double[][] result = new double[size][size];

            if(endRow - startRow <= THRESHOLD){
                for(int i = startRow; i < endRow; i++){
                    for(int j = 0; j < size; j++) {
                        for(int k = 0; k < size; k++){
                            result[i][j] += firstMatrix[i][k] * secondMatrix[k][j];
                        }
                    }
                }
            }else{
                int mid = (startRow + endRow) / 2;
                MatrixMultiplicationTask task1 = new MatrixMultiplicationTask(firstMatrix, secondMatrix, startRow, mid);
                MatrixMultiplicationTask task2 = new MatrixMultiplicationTask(firstMatrix, secondMatrix, mid, endRow);
                invokeAll(task1, task2);
                double[][] result1 = task1.join();
                double[][] result2 = task2.join();

                // Merge results
                for(int i = 0; i < result.length; i++){
                    for(int j = 0; j < result[0].length; j++){
                        if(i < mid) {
                            result[i][j] = result1[i][j];
                        }else{
                            result[i][j] = result2[i - mid][j];
                        }
                    }
                }
            }

            return result;
        }
    }
}
