import numpy as np
from openai import OpenAI
from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility
import json

class MilvusPipeline:
    def __init__(self, client: OpenAI, ebedd_modelname: str, embedding_dim):
        self.host="localhost"
        self.port="19530"
        self.collection = None
        self.embedding_dim = embedding_dim
        # for embedding model
        self.embedding_client = client
        self.ebedd_modelname = ebedd_modelname
        self.connect()

    def connect(self):
        print("*"*100)
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"Connected to Milvus server at {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect to Milvus server: {e}")
            raise

    def create_collection(self, collection_name):        
        # Schema Info
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="related", dtype=DataType.VARCHAR, max_length=512)
        ]
        schema = CollectionSchema(fields=fields, description="Schema for embedding collection")

        # Temporary..
        utility.drop_collection(collection_name)
        # Exsiting Collection
        if utility.has_collection(collection_name):
            self.collection = Collection(collection_name)
            print(f"Collection {collection_name} already exists. Current row count: {self.collection.num_entities}")
            # Schema Update Check
            existing_fields = {field.name: field.dtype for field in self.collection.schema.fields}
            new_fields = {field.name: field.dtype for field in schema.fields}
            
            if not existing_fields == new_fields:
                print(f"Schema has changed.")
                utility.drop_collection(collection_name)
            else:
                return self.collection
        self.collection=Collection(name=collection_name, schema=schema)
        print(f"Collection {collection_name} is newly created.")
        self.ls_collection()
        return self.collection
    
    def insert_data(self, jsonl_file):
        if self.collection is None:
            raise ValueError("Collection is not initialized. Call create_collection first.")
        else :
            if self.collection.num_entities > 1000:
                print(f"insert_data canceled. since {self.collection.num_entities}-data exists")
                return
            try:
                data = []
                with open(jsonl_file, 'r', encoding='utf-8') as file:
                    for line in file:
                        data.append(json.loads(line))
                
                questions = [item['question'] for item in data]
                embeddings = self.get_embedding(questions)

                entities = [
                    {
                        "question": item['question'],
                        "category": json.dumps(item['category'], ensure_ascii=False),
                        "answer": item['answer'],
                        "related": json.dumps(item['related'], ensure_ascii=False),
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
            except Exception as e:
                raise
        
    def create_index(self, field_name):
        if self.collection is None:
            raise ValueError("Collection is not initialized. Call create_collection first.")
        else :
            index_info = self.collection.index().params
            
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 256}
            }

            if index_info["metric_type"] != index_params["metric_type"] or index_info["params"] != index_params["params"] :
                self.collection.release()
                self.collection.drop_index()
            try:
                self.collection.create_index(field_name=field_name, index_params=index_params)
                print(f"Index created on field {field_name}.")
            except Exception as e:
                print(f"Failed to update index parameters")
    
    def load_collection(self):
        self.collection.load()
        print(f"Collection {self.collection.name} loaded.")
        print(f"Current Collection row: {self.collection.num_entities}")

    def ls_collection(self):
        collections = utility.list_collections()
        print("Collection List:")
        for collection in collections:
            print(f"name: {collection}, len: {Collection(collection).num_entities}")

    def close(self):
        connections.disconnect("default")

    def get_embedding(self, texts):
        response = self.embedding_client.embeddings.create(
            model=self.ebedd_modelname,
            input=texts
        )
        print(f"... ({len(texts)})-embedding ...\n  {texts[-1]}")
        return [data.embedding for data in response.data]

    def retrieve_similar_questions(self, prompt, top_k=5, threshold=0.5):
        query_embedding = self.get_embedding([prompt])[0]
        print("-"*80)
        print("Query embedding type:", type(query_embedding), type(query_embedding[0]))
        print("Query embedding shape:", np.array(query_embedding).shape)
        
        search_params = {
            # IP(InnerProduct), L2(Uclidean), Cosine
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding", # target vector
            param=search_params,
            limit=top_k,
            output_fields=["question", "category", "answer", "related"]  # output column
        )
        
        ranked_results = [
            {
                "question": result.entity.get("question"),
                "category": result.entity.get("category"),
                "answer": result.entity.get("answer"),
                "related": result.entity.get("related"),
                "score": 1 / (1 + result.distance)
            }
            for result in results[0]
        ]
        # 결과 출력
        print(f"\nQuery: {prompt}")
        print(f"Top {top_k} similar questions:")
        for i, result in enumerate(ranked_results, 1):
            print(f"\n{i}. Question: {result['question']}")
            print(f"   Answer: {result['answer']}")
            print(f"   Score: {result['score']:.4f}")
        
        filtered_results = [q for q in ranked_results if q['score'] >= threshold]
        print("-------------", len(filtered_results))
        results = filtered_results[:top_k] if len(filtered_results) > top_k else filtered_results
        return results
