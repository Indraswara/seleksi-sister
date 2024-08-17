from Crypto.Cipher import AES
from secrets import token_bytes

ct1 = bytes.fromhex('b6e3251da5c15ebc55f7416aa42ccb879e787425401eb5e6e2ff2572a164b64f5cebd4e9165a3e201b4dc2338928a3b6ac8243f5')  # Replace with actual ct1 hex value
ct2 = bytes.fromhex('b6e23f02a1dd4ab30ab97b5abd70d09bb0483f7d7632bcc1f2c17a509f63d9307cd7cfe736590e081d3dc46fb018a0e9a38856e3f812b2d7af6669d4edd0ab')  # Replace with actual ct2 hex value

leak = b'Shikanoko nokonoko koshitantan! Shikanoko nokonoko koshitantan!'

keystream = bytes([a ^ b for a, b in zip(ct2, leak)])

ct1_flag = bytes([a ^ b for a, b in zip(ct1, keystream)])

print(f'FLAG: {ct1_flag.decode()}')
