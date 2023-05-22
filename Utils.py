def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def find_closest_prime(n):
    while n >= 2:
        if is_prime(n):
            return n
        n -= 1
    return None

n = find_closest_prime(20)
print(n)