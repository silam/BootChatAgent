
https://learn.microsoft.com/en-us/answers/questions/2154792/how-to-fix-similarity-index-was-not-found-for-a-ve

We would like to inform you that, to create vector indexes in Azure Cosmos DB for MongoDB vCore, you need to follow these steps:

Firstly, Use a MongoDB client to connect to your Azure Cosmos DB instance.
Use the create Indexes command to create a vector index. Here is an example of how to do this:
Python

Copy
import pymongo

```
# Connect to MongoDB
client = pymongo.MongoClient("<your_connection_string>")
db = client["your_database"]
collection = db["your_collection"]

# Create the vector index
collection.create_index(
    [("Vector", "cosmosSearch")],
    cosmosSearchOptions={
        "kind": "vector-ivf",
        "numLists": 800,
        "similarity": "COS",
        "dimensions": 1536
    }
)
```


In this example:

"Vector" is the field in your documents that contains the vector data.
"cosmosSearch" specifies that this is a vector index.
cosmosSearchOptions contains the options for the vector index:
"kind" specifies the type of vector index. Options include "vector-ivf" (Inverted File Index) and "vector-hnsw" (Hierarchical Navigable Small Worlds).
"numLists" specifies the number of clusters for the IVF index.
"similarity" specifies the similarity metric, such as "COS" for cosine similarity.
"dimensions" specifies the number of dimensions in the vector.
After creating the index, you can verify that it exists in your collection using below code

Ruby

Copy
indexes = collection.index_information()
for index in indexes:  
    print(index)
Insert your data into the collection. Ensure that the vector data is included in the documents you insert.
Use the $search functionality to perform vector similarity searches. Here is an example of how to perform a vector search:
JavaScript

Copy
query_vector = [0.1, 0.2, 0.3, ...]  # Your query vector
	results = collection.aggregate([
	    {
	        "$search": {
	            "index":"cosmosSearch",
	            "knnBeta": {
	                "vector": query_vector,
	                "path": "Vector",
	                "k": 10
	            }
	        }
	    }
	])
	for result in results:
	    print(result)
In this example:

"Index" is the name of the vector index.
"vector" is the query vector.
"path" is the field in your documents that contains the vector data.
"k" specifies the number of nearest neighbours to return.
By following these steps, you should be able to create vector indexes and perform vector similarity searches in Azure Cosmos DB for MongoDB vCore.

Please refer to the below mentioned links for more information.

https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/vcore/vector-search?tabs=diskann

https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/vcore/indexing

I hope this information helps. Please do let us know if you have any further queries.