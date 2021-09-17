# Discretization
import Orange
iris = Orange.data.Table("iris.tab")
disc = Orange.preprocess.Discretize()
disc.method = Orange.preprocess.discretize.EqualFreq(n=3)
d_iris = disc(iris)

print("Original Dataset:\n")
for e in iris[:3]:
    print(e)
    
print("Discretized Dataset:\n")
for e in d_iris[:3]:
    print(e)

# Continuization
import Orange
titanic = Orange.data.Table("titanic")
continuizer = Orange.preprocess.Continuize()
titanic1 = continuizer(titanic)

print("Before Continuization: ",titanic.domain)
print("After Continuization: ",titanic1.domain)

## Data of 15th Row Before Continuization
print("15th row data before: ",titanic[15])
print("15th row data after: ",titanic1[15])

# Noarmalization
from Orange.data import Table
from Orange.preprocess import Normalize
data = Table("iris")
normalizer = Normalize(norm_type=Normalize.NormalizeBySpan)
normalized_data = normalizer(data)
print("Before Normalization: ",iris[2])
print("After Normalization: ",normalized_data[2])

# Randomization
from Orange.data import Table
from Orange.preprocess import Randomize

data = Table("iris")
randomizer = Randomize(Randomize.RandomizeClasses)
randomized_data = randomizer(data)
print("Before Randomization: ",iris[2])
print("After Randomization: ",randomized_data[2])


