import ast

data = [
    "{'product-id': '03228', 'display-name': 'PetroKing ', 'short-description': \"PetroKing 3228 Men's 6-inch Safety Toe Boot\", 'long-description': 'Extremely lightweight and flexible...', 'brand': 'Red Wing Work', 'manufacturer-name': 'Red Wing Shoes', 'manufacturer-sku': '03228', 'tax-class-id': '100400', 'page-title': '', 'page-description': '', 'image-path': ['img/redwing/64ykfbwetk/300x300px/RW03228C_MUL_N1_0920.jpeg']}",

    "{'product-id': '05705', 'display-name': 'E-Force', 'short-description': \"Men's 9-inch Safety Toe Pull-On Boot\", 'long-description': 'Offering durability, comfort...', 'brand': 'WORX', 'manufacturer-name': 'Red Wing Shoes', 'manufacturer-sku': '05705', 'tax-class-id': '100400', 'page-title': \"Men's E-Force Work Boot 5705 | WORX\", 'page-description': 'This globally certified...', 'image-path': ['img/redwing/wqjew7d9rh/300x300px/WX05705C_MUL_N1_0323.jpeg']}"
]
print(data)

# Convert each string to a Python dict
objects = [ast.literal_eval(item) for item in data]
print(objects)
# Now you can use them as JSON/dict objects
print(objects[0]["product-id"])   # 03228
print(objects[1]["brand"])        # WORX