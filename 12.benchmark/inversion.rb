def get_matrix_inverse(matrix, n)
    augmented = Array.new(n) { Array.new(2 * n, 0.0) }
  
    # Create the augmented matrix [matrix | identity]
    n.times do |i|
      n.times do |j|
        augmented[i][j] = matrix[i][j]
        augmented[i][j + n] = (i == j) ? 1.0 : 0.0
      end
    end
  
    # Perform Gauss-Jordan elimination
    n.times do |i|
      # Make the diagonal contain all 1's
      diag_element = augmented[i][i]
      (2 * n).times do |j|
        augmented[i][j] /= diag_element
      end
  
      # Make the other elements in the current column 0
      n.times do |k|
        next if k == i
        factor = augmented[k][i]
        (2 * n).times do |j|
          augmented[k][j] -= factor * augmented[i][j]
        end
      end
    end
  
    # Extract the inverse matrix from the augmented matrix
    inverse = Array.new(n) { Array.new(n, 0.0) }
    n.times do |i|
      n.times do |j|
        inverse[i][j] = augmented[i][j + n]
      end
    end
  
    inverse
  end
  
  def main
    # Read the size of the matrix
    puts "Enter the size of the matrix: "
    n = gets.to_i
  
    matrix = Array.new(n) { Array.new(n, 0.0) }
  
    # Input each row of the matrix
    puts "Enter the elements of the matrix row by row:"
    n.times do |i|
      row = gets.split.map(&:to_f)
      n.times do |j|
        matrix[i][j] = row[j]
      end
    end
  
    inverse = get_matrix_inverse(matrix, n)
  
    # Print the inverse matrix
    puts "Inverse matrix:"
    inverse.each do |row|
      puts row.map { |val| format('%.6f', val) }.join(' ')
    end
  end
  
  main