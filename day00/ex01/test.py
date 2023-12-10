import numpy as np

from TinyStatistician import TinyStatistician

tstat = TinyStatistician()

a = np.array([1, 42, 300, 10, 59])
print(type(a))
print("np.array([1, 42, 300, 10, 59])")
print("mean:", tstat.mean(a))
# Expected result: 82.4
print("median:", tstat.median(a))
# # Expected result: 42.0
print("quartiles:", tstat.quartile(a))
# # Expected result: [10.0, 59.0]
print("var:", tstat.var(a))
# # Expected result: 12279.439999999999
print("std:", tstat.std(a))
# Expected result: 110.81263465868862

print("[1, 42, 300, 10, 59]")
a = [1, 42, 300, 10, 59]
print("mean:", TinyStatistician().mean(a))
# Output: 82.4
print("median:", TinyStatistician().median(a))
# Output: 42.0
print("quartiles:", TinyStatistician().quartile(a))
# Output: [10.0, 59.0]			# method 1
print("10th percentile:", TinyStatistician().percentile(a, 10))
# Output: 4.6
print("15th percentile:", TinyStatistician().percentile(a, 15))
# Output: 6.4
print("20th percentile:", TinyStatistician().percentile(a, 20))
# Output: 8.2
print("25th percentile:", TinyStatistician().percentile(a, 25))
# Output: 10.0
print("var:", TinyStatistician().var(a))
# Output: 15349.3 				# method 1 Bessel correction
print("std:", TinyStatistician().std(a))
# Output: 123.89229193133849 	# method 1 Bessel correction


b = [1, 42, 300, 10]
print(type(b))
print("[1, 42, 300, 10]")
print("mean:", tstat.mean(b))
print("median:", tstat.median(b))
print("quartiles:", tstat.quartile(b))
print("10th percentile:", tstat.percentile(b, 10))
print("20th percentile:", tstat.percentile(b, 20))
print("var:", tstat.var(b))
print("std:", tstat.std(b))

c = []
print(tstat.mean(c))

d = np.array([])
print(tstat.mean(d))

e = np.array([1, 42, 300, 10, 3, 350, 820])
print(type(e))
print("[1, 42, 300, 10, 3, 350, 820]")
print("mean:", tstat.mean(e))
print("median:", tstat.median(e))
print("quartiles:", tstat.quartile(e))
print("10th percentile:", tstat.percentile(c, 10))
print("20th percentile:", tstat.percentile(c, 20))
print("var:", tstat.var(e))
print("std:", tstat.std(e))

# error tests
print(tstat.mean("test"))
print(tstat.mean(5))
print(tstat.mean((4, 5, 6)))
