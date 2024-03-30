import os
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI

from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    print("Hello LangChain!")

    open_ai_api_key = os.environ["OPENAI_API_KEY"]

    summary_template = """
        Given the information about a person I want you to create:
        1. A short summary
        2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=open_ai_api_key)

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile()

    print(chain.run(information=linkedin_data))
