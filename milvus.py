import numpy as np
from openai import OpenAI
from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility
import json

class MilvusPipeline:
    def __init__(self, client: OpenAI, model_name: str, embedding_dim, collection_name="embedding_collection"):
        # For embddding
        self.embedding_client = client
        self.model_name = model_name
        # For searching
        self.client = connections.connect(host="localhost", port="19530")
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.collection = self.create_collection(collection_name, embedding_dim)

    def create_collection(self, collection_name, dim):
        print("*"*50, "Initialization", "*"*50)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="related", dtype=DataType.VARCHAR, max_length=512)
        ]
        schema = CollectionSchema(fields=fields, description="Schema for embedding collection")
        if utility.has_collection(collection_name):
            # Collection check
            existing_collection = Collection(collection_name)
            existing_schema = existing_collection.schema
            existing_fields = {field.name: field.dtype for field in existing_schema.fields}
            new_fields = {field.name: field.dtype for field in schema.fields}
            
            if existing_collection.num_entities > 3000: #Temp while test
                print(f"Data .")
                utility.drop_collection(collection_name)
            elif not existing_fields == new_fields:
                print(f"Schema has changed.")
                utility.drop_collection(collection_name)
            else:
                print(f"Collection {collection_name} already exists with matching schema.")
                print(f"Existing Collection row: {existing_collection.num_entities}")
                return existing_collection
        print(f"Collection {collection_name} is newly created.")
        return Collection(name=collection_name, schema=schema)
    
    def insert_data(self, jsonl_file):
        if self.collection.num_entities :
            print(f"data insert has been canceled. {self.collection.num_entities}-data exist")
            return
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        
        questions = [item['question'] for item in data]
        embeddings = self.get_embedding(questions)

        entities = [
            {
                "question": item['question'],
                "category": json.dumps(item['category']),
                "answer": item['answer'],
                "related": json.dumps(item['related']),
                "embedding": embedding
            }
            for item, embedding in zip(data, embeddings)
        ]

        #print(f"Embeddings type: {type(embeddings)}")
        #print(f"First embedding type: {type(embeddings[0])}")
        #print(f"First embedding length: {len(embeddings[0])}")
        #print(self.collection.schema)
            
        self.collection.insert(entities)
        print(f"Inserted {len(entities)} entities.")
        
    def create_index(self, field_name):
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.collection.create_index(field_name=field_name, index_params=index_params)
        print(f"Index created on field {field_name}.")
    
    def load_collection(self):
        self.collection.load()
        print(f"Collection {self.collection.name} loaded.")
        print(f"Current Collection row: {self.collection.num_entities}")

    def check_collections(self):
        collections = self.client.list_collections()
        print("Currnt Collections:")
        for collection in collections:
            print(collection)

    def get_embedding(self, texts):
        print("... embedding for ...")
        response = self.embedding_client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        print(f"... embedding is done for {len(texts)} questions ...")
        return [data.embedding for data in response.data]

    def retrieve_similar_questions(self, prompt, n=5):
        query_embedding = self.get_embedding([prompt])[0]
        print("-"*200)
        print("Query embedding type:", type(query_embedding))
        print("Query embedding shape:", np.array(query_embedding).shape)
        print("First few values of query embedding:", query_embedding[:5])
        
        search_params = {
            # IP(InnerProduct), L2(Uclidean), Cosine
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding", # target vector
            param=search_params,
            limit=n,
            output_fields=["question", "category", "answer", "related"]  # output column
        )
        
        ranked_results = [
            {
                "question": result.entity.get("question"),
                "category": result.entity.get("category"),
                "answer": result.entity.get("answer"),
                "related": result.entity.get("related"),
                "distance": result.distance
            }
            for result in results[0]
        ]
        return ranked_results
