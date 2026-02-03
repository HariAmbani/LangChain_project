from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from autochain import AutoChain
from dotenv import load_dotenv
from langchain.agents import initialize_agent, load_tools, AgentType, Tool

load_dotenv()

def generate_pet_name(animal_type, pet_color):
    llm = OpenAI(temperature=0.7)
    
    prompt_template_name = PromptTemplate(
        input_variables=['animal_type', "pet_color"],
        template="Suggest three unique and cool pet names for a {animal_type} with the color {pet_color}." 
    )
    
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="pet_names")
    
    response = name_chain({"animal_type": animal_type, "pet_color": pet_color})
    return response

def langchain_agent():
    llm = OpenAI(temperature=0.5)
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools = tools,
        llm = llm,
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True
    )
    result = agent.run("What is the average age of the dog ? and what is the square root of it ?")
    print(result)

if __name__ == "__main__":
    names = generate_pet_name("cow", "brown")
    print("Suggested Pet Names: ")
    print(names)
