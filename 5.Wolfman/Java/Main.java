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

            /**
             * ForkJoinPool adalah class yang digunakan untuk mengatur task yang akan dijalankan
             * ForkJoinPool memiliki method invoke yang digunakan untuk menjalankan task
             * 
             */
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
                continue;
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
        private static final int THRESHOLD = 2;
        private double[][] firstMatrix;
        private double[][] secondMatrix;
        private int startRow;
        private int endRow;

        /**
         * 
         * @param firstMatrix
         * @param secondMatrix
         * @param startRow
         * @param endRow
         * 
         * Constructor dari class MatrixMultiplicationTask
         * class ini ada di dalam class MatrixMultiplicationTask
         */
        public MatrixMultiplicationTask(double[][] firstMatrix, double[][] secondMatrix, int startRow, int endRow){
            this.firstMatrix = firstMatrix;
            this.secondMatrix = secondMatrix;
            this.startRow = startRow;
            this.endRow = endRow;
        }
        
        /**
         * 
         * @return double[][]
         * compute adalah method yang menghitung perkalian matriks
         * 
         */
        @Override
        protected double[][] compute(){
            int size = firstMatrix.length;
            double[][] result = new double[size][size];
            
            /**
             * jika ukuran matriks lebih kecil atau sama dengan threshold
             * maka matriks akan dihitung secara langsung
             * tanpa dibagi menjadi dua bagian
             */
            if(endRow - startRow <= THRESHOLD){
                for(int i = startRow; i < endRow; i++){
                    for(int j = 0; j < size; j++) {
                        for(int k = 0; k < size; k++){
                            result[i][j] += firstMatrix[i][k] * secondMatrix[k][j];
                        }
                    }
                }
            }else{
                /**
                 * jika ukuran matriks lebih besar dari threshold
                 * maka matriks akan dibagi menjadi dua bagian
                 */
                int mid = (startRow + endRow) / 2;
                MatrixMultiplicationTask task1 = new MatrixMultiplicationTask(firstMatrix, secondMatrix, startRow, mid);
                MatrixMultiplicationTask task2 = new MatrixMultiplicationTask(firstMatrix, secondMatrix, mid, endRow);
                invokeAll(task1, task2);
                double[][] result1 = task1.join();
                double[][] result2 = task2.join();

                /**
                 * penggabung hasil dari dua task yang sudah dijalankan 
                 * task1 dan task2 
                 * 
                 * task1 menghitung hasil perkalian matriks dari baris 0 sampai mid
                 * task2 menghitung hasil perkalian matriks dari baris mid sampai endRow
                 * 
                 * hasil dari task1 dan task2 digabungkan menjadi satu matriks
                 * 
                 * hasil dari task1 disimpan di result1
                 * hasil dari task2 disimpan di result2
                 */
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
