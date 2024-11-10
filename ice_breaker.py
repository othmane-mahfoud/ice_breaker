from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

from dotenv import load_dotenv
import os

def ice_break_with(name: str) -> str:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)
    
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
    
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url = 'https://www.linkedin.com/in/othmane-mahfoud/'
    )
    
    res = chain.invoke(input={"information": linkedin_data})
    
    print(res)
    

if __name__ == '__main__':
    
    load_dotenv(override=True)
    ice_break_with(name="Eden Marco")