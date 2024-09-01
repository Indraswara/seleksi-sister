import requests
import pyotp
import base64
import json
import hashlib

# Informasi pengguna
NIM = '13522119'
waifu = 'Violet'
shared_secret = f'seleksister24{NIM}{waifu}'

# Konfigurasi TOTP
interval = 30  # Time step
algorithm = 'SHA256'
digits = 8  # Jumlah digit OTP

# Membuat secret key untuk TOTP
secret = base64.b32encode(shared_secret.encode('utf-8')).decode('utf-8')

# Membuat TOTP dengan pyotp
totp = pyotp.TOTP(secret, interval=interval, digits=digits, digest=hashlib.sha256)
otp = totp.now()

# Membuat kredensial Authorization
auth = f'{NIM}:{otp}'
auth_bytes = auth.encode('utf-8')
auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')
authorization_header = f'Basic {auth_base64}'

# Membuat payload JSON
payload = {
    "fullname": "Indraswara Galih Jayanegara",
    "link": "https://github.com/Indraswara/seleksi-sister",
    "message": "Kadang rasanya seperti tenggelam dalam lautan tugas yang tiada akhir. Setiap kali satu selesai, datang lagi yang lain, seolah-olah tak ada waktu untuk sekadar bernapas. Mata lelah menatap layar, pikiran terus berputar mencoba mengurai segala beban, namun hati ini mulai letih. Rasanya ingin menyerah, tapi tahu kalau itu bukan pilihan. Hanya bisa berharap waktu berjalan lebih cepat, dan akhirnya bisa melihat cahaya di ujung terowongan ini."

}

# Melakukan request POST
url = 'http://sister21.tech:7787/recruitment/submit/b'
headers = {
    'Content-Type': 'application/json',
    'Authorization': authorization_header
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Menampilkan hasil
    print(f'Status Code: {response.status_code}')
    print(f'Response: {response.text}')
    print(f'Headers: {response.headers}')
except requests.exceptions.RequestException as e:
    print(f'Error: {e}')
