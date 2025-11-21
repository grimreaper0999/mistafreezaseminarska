from math import floor, gcd, log

N = int(input("num "))
n = floor(log(N - 1, 2)) + 1
print(f"num bits: {n}")

a = int(input("a value "))

def tobin(z):
  z_ = z
  d = 2**(n-1)
  b = ""
  for i in range(n):
    #print(f"{b}, {z_}, {d}, {z_//d}")
    b += str(int(z_//d))
    z_ -= d*(z_//d)
    d /= 2
  return b

print("\n".join([f"{tobin(2**i)} -> {tobin(((2**i)*a)%N)}" for i in range(n)]))