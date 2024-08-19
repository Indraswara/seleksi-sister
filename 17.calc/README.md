# Kalkulator Sederhana Perl

Skrip ini merupakan implementasi kalkulator sederhana yang mendukung operasi aritmatika dasar seperti penjumlahan, pengurangan, perkalian, pembagian, serta pangkat. Semua operasi dilakukan menggunakan bitwise operation tanpa menggunakan operator aritmatika bawaan.

## Fitur yang Didukung
- **Penjumlahan (`+`)**
- **Pengurangan (`-`)**
- **Perkalian (`*`)**
- **Pembagian (`/`)**
- **Pangkat (`^`)**

## Fungsi Utama

### `increment($n)`
Meningkatkan nilai `$n` tanpa menggunakan operator penambahan.

### `decrement($n)`
Menurunkan nilai `$n` tanpa menggunakan operator pengurangan.

### `negate($n)`
Mengembalikan nilai negatif dari `$n`.

### `my_abs($n)`
Mengembalikan nilai absolut dari `$n`.

### `add($a, $b)`
Menambahkan dua bilangan `$a` dan `$b` menggunakan operasi bitwise.

### `subtract($a, $b)`
Mengurangi `$b` dari `$a` menggunakan operasi bitwise.

### `power($base, $exponent)`
Menghitung nilai `$base` pangkat `$exponent` menggunakan operasi bitwise.

### `multiply($a, $b)`
Mengalikan dua bilangan `$a` dan `$b` menggunakan operasi bitwise.

### `divide($a, $b)`
Membagi `$a` dengan `$b` menggunakan operasi bitwise. Akan mengembalikan kesalahan jika pembagian dengan nol dilakukan.

## Cara Penggunaan
1. Jalankan skrip.
2. Masukkan ekspresi aritmatika menggunakan operator yang didukung, misalnya:
    ```
    Masukkan operasi: 5 + 3 * 2 ^ 2
    ```
   Ekspresi ini akan dipecah menjadi operasi berurutan.
3. Ketik `exit` untuk keluar dari kalkulator.

## Contoh Penggunaan
```plaintext
Kalkulator sederhana
untuk exit ketik 'exit'
Masukkan operasi: 5 + 3 * 2 ^ 2
Result: 17
Masukkan operasi: exit
