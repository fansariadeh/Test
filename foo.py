# Suppose this is foo.py.

print("before import")
import math

print("before function_a")
def function_a():
    print("Function A")

print("before function_b")
def function_b():
    print("Function B {}".format(math.sqrt(100)))

print("before __name__ guard")
if __name__ == '__main__':
    function_a()
    function_b()
print("after __name__ guard")

# # What gets printed if foo is the main program

# before import
# before function_a
# before function_b
# before __name__ guard
# Function A
# Function B 10.0
# after __name__ guard
#--------------------------------------------------------------------------------------------------------
# # What gets printed if foo is imported as a regular module

# before import
# before function_a
# before function_b
# before __name__ guard
# after __name__ guard