list_of_dicts = [
    {'id': 0, 'embed': [0.4937723169626643, 0.43335433190404127, 0.8187348585930836, 0.34622322564002284, 0.7649056106341516, 0.6889160617701864, 0.27486782593025993]},
    {'id': 1, 'embed': [0.12, 0.56, 0.78, 0.23, 0.98, 0.45, 0.67]},
    {'id': 2, 'embed': [0.77, 0.34, 0.21, 0.89, 0.11, 0.56, 0.34]}
]

set_of_frozensets = set()

for dictionary in list_of_dicts:
    # Convert the 'embed' list to a tuple
    embed_tuple = tuple(dictionary['embed'])
    frozenset_dict = frozenset([('id', dictionary['id']), ('embed', embed_tuple)])
    set_of_frozensets.add(frozenset_dict)

# Accessing elements in the set
for frozenset_item in set_of_frozensets:
    dict_item = dict(frozenset_item)
    print(f"ID: {dict_item['id']}, Embed: {dict_item['embed']}")
