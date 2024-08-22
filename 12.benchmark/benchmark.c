#include <stdio.h>
#include <stdlib.h>

void run_benchmark(const char* command, const char* lang, const char* input_file, FILE* output_file) {
    printf("Benchmarking %s...\n", lang);
    char full_command[512];
    snprintf(full_command, sizeof(full_command), "/usr/bin/time -f '%%e' -o temp_time.txt %s < %s", command, input_file);
    system(full_command);

    FILE* file = fopen("temp_time.txt", "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    char buffer[128];
    if (fgets(buffer, sizeof(buffer), file) != NULL) {
        fprintf(output_file, "%s execution time: %sseconds\n", lang, buffer);
    }

    fclose(file);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    const char* input_file = argv[1];
    FILE* output_file = fopen("benchmark_times.txt", "w");
    if (output_file == NULL) {
        perror("Error opening output file");
        return 1;
    }

    run_benchmark("java ./code/inversion.java", "Java", input_file, output_file);
    run_benchmark("php ./code/inversion.php", "PHP", input_file, output_file);

    //running cpp 
    system("g++ -o ./code/inversion_cpp ./code/inversion.cpp");
    run_benchmark("./code/inversion_cpp", "C++", input_file, output_file);

    system("rustc -o ./code/inversion_rust ./code/inversion.rs");
    run_benchmark("./code/inversion_rust", "Rust", input_file,output_file);
    run_benchmark("ruby ./code/inversion.rb", "Ruby", input_file,output_file);
    run_benchmark("python3 ./code/inversion.py", "Python", input_file,output_file);
    run_benchmark("perl ./code/inversion.pl", "Perl", input_file, output_file);
    run_benchmark("node ./code/inversion.js", "JavaScript", input_file, output_file);

    //running go 
    system("go build -o ./code/inversion_go ./code/inversion.go");
    run_benchmark("./code/inversion_go", "Go", input_file, output_file);

    //running c
    system("gcc -o ./code/inversion_c ./code/inversion.c");
    run_benchmark("./code/inversion_c", "C", input_file, output_file);

    fclose(output_file);

    system("rm -f temp_time.txt");
    return 0;
}