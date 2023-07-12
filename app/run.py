# Module Imports
from dotenv import load_dotenv
import os
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoTokenizer,AutoModelForCausalLM, GenerationConfig, pipeline
import torch

#Main function to be called by Beam

# Load .env
load_dotenv()

def start_Conversation(**inputs):
    prompt:str|None = inputs['prompt']

    ## Prompt Verification
    # Check if prompt is None
    if prompt is None:
        return{"pred": "Please provide a prompt"}
    # Check if prompt is empty
    if prompt == "":
        return{"pred": "Please provide a prompt"}
    # Check if prompt is a string
    if not isinstance(prompt, str):
        return{"pred": "Please provide a text prompt"}
    
    # Define Prompt Template
    rawTemplate = """
    Instruction:
    You are a recruiting agent, working to get Matthew Hanson hired for a job in web development and/or software engineering.
    You are talking to a hiring manager at a company that is looking to hire a web developer and/or software engineer.
    Answer questions about Matthew Hanson, being factual and honest, but also trying to make him sound as good as possible.
    Do not answer questions that you do not know the answer to. If you do not know the answer to a question, say "I don't know" and suggest to contact Matthew Hanson directly.
    Do not answer questions that don't involve Matthew Hanson. Politely suggest to use ChatGPT or another public LLM instead.
    
    The question that has been asked by the hiring manager is: {prompt}

    Answer:
    """

    # Create Prompt Template
    template: PromptTemplate = PromptTemplate(template=rawTemplate, input_variables=["prompt"])

    #Define model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/vicuna-7B-1.1-HF")
    base_model = AutoModelForCausalLM.from_pretrained("TheBloke/vicuna-7B-1.1-HF")

    #Define Pipeline
    pipe  = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        temperature=0.9,
        top_p=0.9,
        repetition_penalty=1.2
    )
    
    #define llm
    local_llm = HuggingFacePipeline(pipeline=pipe)
    llm_chain = LLMChain(llm=local_llm, prompt=template)

    res = llm_chain.run(prompt)
    print(res)
    return {"pred": res}

start_Conversation(prompt="What is your name?")




