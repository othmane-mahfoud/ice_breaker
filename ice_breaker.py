from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

information = """
Elon Reeve Musk FRS (/ˈiːlɒn/; born June 28, 1971) is a businessman and investor known for his key roles in the space company SpaceX and the automotive company Tesla, 
Inc. Other involvements include ownership of X Corp., the company that operates the social media platform X (formerly Twitter), and his role in the founding of the 
Boring Company, xAI, Neuralink, and OpenAI. He is one of the wealthiest individuals in the world; as of November 2024 Forbes estimates his net worth to be US$290 
billion.
"""

if __name__ == '__main__':
    
    load_dotenv(override=True)
    
    summary_template = """
        given the Linkedin information {information} about a person I want you to create:
        1. A short summary
        2. two interesting facts about them
    """
    
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    
    # llm = ChatOpenAI(
    #     temperature=0, 
    #     model="gpt-3.5-turbo", 
    #     api_key=os.environ['OPENAI_API_KEY']
    # )
    
    llm = ChatOllama(model="llama3")
    
    chain = summary_prompt_template | llm | StrOutputParser()
    
    res = chain.invoke(input={"information": information})
    
    print(res)