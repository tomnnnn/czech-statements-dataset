from .dataset import Dataset
import time

dataset = Dataset("datasets/curated.sqlite", read_only=True)

timestamp = time.time() 
print(dataset.get_segments())

print("Time taken to get segments:", time.time() - timestamp)
