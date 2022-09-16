import hashlib


def gen_hash():
    print("==================================================★ HASH256 ★==================")
    put = input("encrypt keyword 입력 : ")
    passwd = ''
    result = hashlib.sha256(put.encode())
    passwd += result.hexdigest()
    print("Start keyword : ", put)

    print("==================================================★ WAIT ... ★================")

    num = int(input("반복 횟수 : "))
    for i in range(1, num+1):
        result = hashlib.sha256(passwd.encode())
        print("\n create hash(%d) : " % i, passwd)
        print("SHA256 : ", result.hexdigest())

        passwd = ''
        passwd += result.hexdigest()


def is_valid(itr: int, value: str, key: str):
    value = '@' + value + '@'
    temp_value = ''
    result = hashlib.sha256(value.encode())
    temp_value += result.hexdigest()

    for i in range(0, itr):
        result = hashlib.sha256(temp_value.encode())
        temp_value = ''
        temp_value += result.hexdigest()

    return temp_value == key


if __name__ == '__main__':
    gen_hash()
