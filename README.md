# Hands-On AI: Building and Deploying LLM-Powered Apps
This is the repository for the LinkedIn Learning course `Hands-On AI: Building and Deploying LLM-Powered Apps`. The full course is available from [LinkedIn Learning][lil-course-url].

_See the readme file in the main branch for updated instructions and information._
## Lab2: Adding LLM to Chainlit App
Now we have a web interface working, we will now add an LLM to our Chainlit app to have our simplified version of ChatGPT. We will be using [Langchain](https://python.langchain.com/docs/get_started/introduction) as the framework for this course. It provides easy abstractions and a wide varieties of data connectors and interfaces for everything LLM app development.

In this lab, we will be adding an Chat LLM to our Chainlit app using Langchain.

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

- [Langchain's Prompt Template](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/#chatprompttemplate)
- [Langchain documentation](https://python.langchain.com/docs/modules/chains/foundational/llm_chain#legacy-llmchain)
- [Chainlit's documentation](https://docs.chainlit.io/get-started/pure-python)
