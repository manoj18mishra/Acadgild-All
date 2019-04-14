import random
class Cipher:
    cipher_key = random.randint(1,50)
    #cipher_key = 8
    def __init__(self):
        self.text = input("Please enter the text to be encrypted:")
        print("The key is: {}".format(str(self.cipher_key)))
        self.encrypt()
        self.decrypt()
    def encrypt(self):
        self.e_text = ''.join((chr(ord(x)+self.cipher_key) for x in self.text if x.isalnum()))
        print("The encrypted text is: ", self.e_text)
    def decrypt(self):
        self.e_text = ''.join((chr(ord(x)-self.cipher_key) for x in self.e_text))
        print("The decrypted text is: {}".format(self.e_text))

c1=Cipher()