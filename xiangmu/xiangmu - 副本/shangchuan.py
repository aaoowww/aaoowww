from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os
from keys import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENV
from langchain.vectorstores import Pinecone

'''
下面的这部分代码是将文件夹中的word文档，上传到自己的向量数据库
'''

# # 设置代理URL
# import os
# os.environ["OPENAI_BASE_URL"] = "api.openai-proxy.com"

directory_path = 'test'  # 这边填入你自己的数据文件所在的文件夹
data = []
# loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".doc") or filename.endswith(".docx"):
        # print the file name
        # langchain自带功能，加载word文档
        loader = UnstructuredWordDocumentLoader(f'{directory_path}/{filename}')
        print(loader)
        data.append(loader.load())
print(len(data))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
texts = []
for i in range(len(data)):
    print(i)
    texts.append(text_splitter.split_documents(data[i]))
    print(text_splitter.split_documents(data[i]))
print(len(texts))

# Creating embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index_name = "dental"  # put in the name of your pinecone index here
for i in range(len(texts)):
    Pinecone.from_texts([t.page_content for t in texts[i]], embeddings, index_name=index_name)
    print("done")
