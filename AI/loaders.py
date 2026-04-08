import os
from langchain_community.document_loaders import TextLoader


class DocumentLoader:

    def load_documents(self,DATA_PATH):
        documents = []
        

        for file in os.listdir(DATA_PATH):
            path = os.path.join(DATA_PATH, file)

            loader = TextLoader(path)
            docs = loader.load()

            for d in docs:
                d.metadata = {"source": file}

            documents.extend(docs)

        return documents