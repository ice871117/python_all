map = dict()

def test_index_map(key):
    if map.get(key) == None:
        map[key] = len(map)


test_index_map("a")
test_index_map("b")
test_index_map("a")
test_index_map("c")
test_index_map("d")
test_index_map("e")
test_index_map("c")
test_index_map("d")
test_index_map("f")

print(map)