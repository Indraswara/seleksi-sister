#!/usr/bin/perl
use strict;
use warnings;

sub increment{
    my ($n) = @_;
    increment_loop:
    {
        if (($n & 1) == 0) {
            $n = $n | 1;
            goto increment_loop;
        } else {
            $n = $n >> 1;
        }
    }
    return $n;
}

sub decrement{
    my ($n) = @_;
    decrement_loop:
    {
        if (($n & 1) == 1) {
            $n = $n & ~1;
            goto decrement_loop;
        } else {
            $n = $n >> 1;
        }
    }
    return $n;
}

sub negate{
    my ($n) = @_;
    my $negated = subtract(0, $n);
    return $negated;
}

sub my_abs{
    my ($n) = @_;
    return $n < 0 ? negate($n) : $n;
}

sub add{
    my ($a, $b) = @_;
    my $carry;
ADD_LOOP:
    if ($b != 0) {
        $carry = $a & $b;
        $a = $a ^ $b;
        $b = $carry << 1;
        goto ADD_LOOP;
    }
    return $a;
}

sub subtract{
    my ($a, $b) = @_;
    
    $b = ~$b;
    my $carry = 1;

    convert_to_twos_complement:
    {
        my $sum = $b ^ $carry;
        $carry = ($b & $carry) << 1;
        $b = $sum;
        goto convert_to_twos_complement if $carry;
    }

    my $result = $a;
    $carry = $b;

    add:
    {
        my $sum = $result ^ $carry;
        $carry = ($result & $carry) << 1;
        $result = $sum;
        goto add if $carry;
    }

    $result = unpack('q', pack('q', $result));
    return $result;
}

sub power {
    my ($base, $exponent) = @_;
    my $result = 1;

    power_loop:
    {
        if ($exponent & 1) {
            $result = multiply($result, $base);
        }
        $base = multiply($base, $base);
        $exponent = $exponent >> 1;
        if ($exponent > 0) {
            goto power_loop;
        }
    }

    return $result;
}

sub multiply {
    my ($a, $b) = @_;
    my $result = 0;
    my $positive = 1;

    if($a < 0){
        $a = negate($a);
        $positive = !$positive;
    }
    if($b < 0){
        $b = negate($b);
        $positive = !$positive;
    }

    my $a_copy = $a;
    my $b_copy = $b;

    multiply_loop:
    {
        if($b_copy > 0){
            if (($b_copy & 1) == 1) {
                $result = add($result, $a_copy);
            }
            $a_copy = $a_copy << 1;
            $b_copy = $b_copy >> 1;
            goto multiply_loop;
        }
    }

    return $positive ? $result : negate($result);
}

sub divide{
    my ($a, $b) = @_;
    die "Division by zero error" if $b == 0;

    my $result = 0;
    my $positive = 1;

    if($a < 0){
        $a = negate($a);
        $positive = !$positive;
    }
    if($b < 0){
        $b = negate($b);
        $positive = !$positive;
    }

    my $temp_a = $a;
    my $divisor = $b;

    divide_loop:
    {
        if($temp_a >= $divisor){
            $temp_a = subtract($temp_a, $divisor);
            $result = add($result, 1);
            goto divide_loop;
        }
    }

    return $positive ? $result : negate($result);
}

print "Kalkulator sederhana\n";
print "untuk exit ketik 'exit'\n";

START:
print "Masukkan operasi: ";
my $expression = <STDIN>;
chomp($expression);
if($expression eq 'exit'){
    goto END;
}
my @tokens = split / /, $expression;
my $result = int(shift @tokens);  # Convert to integer

OPERATION_LOOP:
if(@tokens){
    my $operator = shift @tokens;
    my $operand = int(shift @tokens);  # Convert to integer
    
    if($operator eq '+'){
        $result = add($result, $operand);
    }elsif($operator eq '-'){
        $result = subtract($result, $operand);
    }elsif($operator eq '*'){
        $result = multiply($result, $operand);
    }elsif($operator eq '/'){
        $result = divide($result, $operand);
    }elsif($operator eq '^'){
        $result = power($result, $operand);
    }
    else{
        die "Unknown operator: $operator";
    }
    
    goto OPERATION_LOOP;
}
print "Result: $result\n";
goto START;

END: