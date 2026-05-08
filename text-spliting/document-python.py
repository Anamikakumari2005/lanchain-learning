from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid   # element mil gaya

        elif arr[mid] < target:
            left = mid + 1   # right side jao

        else:
            right = mid - 1  # left side jao

    return -1   # element nahi mila


# Example
arr = [1, 3, 5, 7, 9, 11]
target = 7

result = binary_search(arr, target)

if result != -1:
    print("Element found at index:", result)
else:
    print("Element not found")
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=400,
    chunk_overlap=0,
)

chunk = splitter.split_text(text)

print(len(chunk))
print(chunk[1])