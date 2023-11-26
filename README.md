# Hands-On AI: Building and Deploying LLM-Powered Apps
This is the repository for the LinkedIn Learning course `Hands-On AI: Building and Deploying LLM-Powered Apps`. The full course is available from [LinkedIn Learning][lil-course-url].

_See the readme file in the main branch for updated instructions and information._
## Lab5: Putting it All Together
In Lab 2, we created the basic scaffold of our Chat with PDF App. In Lab 3, we added PDF uploading and processing functionality. In Lab 4, we added the capability to indexing documents into a vector database. Now we have all the required pieces together, it's time for us to assemble our RAG (retrieval-augmented generation) system using Langchain.


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
