use strict;
use warnings;

sub print_matrix {
    my ($matrix) = @_;
    for my $row (@$matrix) {
        print join(" ", @$row), "\n";
    }
}

sub transpose_matrix {
    my ($matrix) = @_;
    my $size = scalar @$matrix;
    my @transposed;
    for my $i (0 .. $size - 1) {
        for my $j (0 .. $size - 1) {
            $transposed[$j][$i] = $matrix->[$i][$j];
        }
    }
    return \@transposed;
}

sub get_matrix_minor {
    my ($matrix, $row, $col) = @_;
    my $size = scalar @$matrix;
    my @minor;
    for my $i (0 .. $size - 1) {
        next if $i == $row;
        my @minor_row;
        for my $j (0 .. $size - 1) {
            next if $j == $col;
            push @minor_row, $matrix->[$i][$j];
        }
        push @minor, \@minor_row;
    }
    return \@minor;
}

sub get_matrix_determinant {
    my ($matrix) = @_;
    my $size = scalar @$matrix;
    return $matrix->[0][0] if $size == 1;
    return $matrix->[0][0] * $matrix->[1][1] - $matrix->[0][1] * $matrix->[1][0] if $size == 2;

    my $det = 0;
    for my $i (0 .. $size - 1) {
        my $minor = get_matrix_minor($matrix, 0, $i);
        $det += ((-1) ** $i) * $matrix->[0][$i] * get_matrix_determinant($minor);
    }
    return $det;
}

sub get_matrix_inverse {
    my ($matrix) = @_;
    my $size = scalar @$matrix;
    my $det = get_matrix_determinant($matrix);
    die "Matrix is singular and cannot be inverted" if $det == 0;

    if ($size == 2) {
        return [
            [$matrix->[1][1] / $det, -$matrix->[0][1] / $det],
            [-$matrix->[1][0] / $det, $matrix->[0][0] / $det]
        ];
    }

    my @cofactors;
    for my $r (0 .. $size - 1) {
        my @cofactor_row;
        for my $c (0 .. $size - 1) {
            my $minor = get_matrix_minor($matrix, $r, $c);
            push @cofactor_row, ((-1) ** ($r + $c)) * get_matrix_determinant($minor);
        }
        push @cofactors, \@cofactor_row;
    }

    my $transposed_cofactors = transpose_matrix(\@cofactors);
    for my $r (0 .. $size - 1) {
        for my $c (0 .. $size - 1) {
            $transposed_cofactors->[$r][$c] /= $det;
        }
    }

    return $transposed_cofactors;
}

sub main {
    print "Enter the size of the matrix: ";
    my $n = <STDIN>;
    chomp $n;

    my @matrix;
    print "Enter the matrix row by row:\n";
    for my $i (0 .. $n - 1) {
        my $line = <STDIN>;
        chomp $line;
        my @row = split ' ', $line;
        push @matrix, \@row;
    }

    eval {
        my $inverse = get_matrix_inverse(\@matrix);
        print "The inverse of the matrix is:\n";
        print_matrix($inverse);
    };
    if ($@) {
        print "Error: $@\n";
    }
}

main();