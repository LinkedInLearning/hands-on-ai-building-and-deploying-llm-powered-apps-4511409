# Hands-On AI: Building and Deploying LLM-Powered Apps
This is the repository for the LinkedIn Learning course `Hands-On AI: Building and Deploying LLM-Powered Apps`. The full course is available from [LinkedIn Learning][lil-course-url].

_See the readme file in the main branch for updated instructions and information._
## Lab4: Indexing Documents into Vector Database
In the previous lab, we enabled document loading and chunking them into smaller sub documents. Now, we will need to index them into our search engine vector databse in order for us to build our Chat with PDF application using the RAG (Retrieval Augmented Generation) pattern.

In this lab, we will implement adding OpenAI's embedding model and index the documents we chunked in the previous section into a Vector Database. We will be using [Chroma](https://www.trychroma.com/) as the vector database of choice. Chroma is a lightweight embedding database that can live in memory, similar to SQLite.

## Exercises

We will build on top of our existing chainlit app code in `app/app.py` in the `app` folder. As in our previous app, we added some template code and instructions in `app/app.py`

1. Please go through the exercises in `app/app.py`. 

2. Please lanuch the application by running the following command on the Terminal:

```bash
chainlit run app/app.py -w
```

## Solution

Please see `app/app.py`.

Alternatively, to launch the application, please run the following command on the Terminal:

```bash
chainlit run app/app.py -w
```


## References

- [Langchain Embedding Models](https://python.langchain.com/docs/modules/data_connection/text_embedding/)
- [ChromaDB Langchain Integration](https://docs.trychroma.com/integrations/langchain)
