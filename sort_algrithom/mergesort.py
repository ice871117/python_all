def mergeTwoArr(arr, start, middle, end):
	res = []
	i = start
	j = middle
	while i < middle and j < end:
		if arr[i] < arr[j]:
			res.append(arr[i])
			i += 1
		else:
			res.append(arr[j])
			j += 1
	while i < middle:
		res.append(arr[i])
		i += 1
	while j < end:
		res.append(arr[j])
		j += 1
	arr[start:end] = res

def mergeSort(arr, start, end):
	if end - start <= 1:
		return
	middle = int((start + end)/2)
	mergeSort(arr, start, middle)
	mergeSort(arr, middle, end)
	mergeTwoArr(arr, start, middle, end)


test = [3, 10, 9, 4, 7, 8, 5, 6, 2, 1]
mergeSort(test, 0, len(test))
print(test)