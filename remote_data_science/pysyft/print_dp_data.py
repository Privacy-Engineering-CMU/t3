dataset = ds_node.datasets[8]
dataset

results = [dataset['Rating_Data'] for i in range(10)]

print(results)

from time import sleep
for result in results:
    ptr = result.publish()
    print(result)
    print(ptr)
    sleep(1)