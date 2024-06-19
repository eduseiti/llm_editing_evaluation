from groq import Groq

import time
import json


GROQ_LLAMA3_70B_MODEL="llama3-70b-8192"


#
# Prompt for factual association extraction from a given text
#

FACTUAL_ASSOCIATIONS_EXTRACTION_SYSTEM=(
    "You read a text and break it down in a sequence of factual associations sentences."
)

FACTUAL_ASSOCIATIONS_EXTRACTION_PROMPT=(
    "Read the text and return a list of all factual associations you can "
    "extract exclusively from it. Write sentences which are self contained "
    "and includes the maximum information provided, including the implicit "
    "ones and temporal information. For each factual association, identify "
    "the subject, the relation and the object. Only output the JSON format, "
    "nothing else: "
    "{\"sentences\":[{\"subject\":\"<subject-1>\", "
                     "\"relation\":\"<relation-1>\", "
                     "\"object\":\"object-1\"}, ..., "
                    "{\"subject\":\"<subject-n>\", "
                     "\"relation\":\"<relation-n>\", "
                     "\"object\":\"object-n\"}]}"
)

FACTUAL_ASSOCIATIONS_EXTRACTION_TEXT_TEMPLATE="\n\nText: \"{}\""



#
# Prompt for simple factual association extraction from a given text
#

SIMPLE_FACTUAL_ASSOCIATIONS_EXTRACTION_SYSTEM=(
    "You read a text and break it down in a sequence of factual associations sentences."
)

SIMPLE_FACTUAL_ASSOCIATIONS_EXTRACTION_PROMPT=(
    "Read the text and return a list of all simple factual associations you "
    "can extract exclusively from it. Write independent sentences also including "
    "the implicit and temporal information. Break down the information in "
    "sentences containing a single object. For each factual association, identify "
    "the subject, the relation and the object. Only output the JSON format, "
    "nothing else before or after: "
    "{\"sentences\":[{\"subject\":\"<subject-1>\", "
                     "\"relation\":\"<relation-1>\", "
                     "\"object\":\"<object-1>\"}, ..., "
                    "{\"subject\":\"<subject-n>\", "
                     "\"relation\":\"<relation-n>\", "
                     "\"object\":\"<object-n>\"}]}"
)

SIMPLE_FACTUAL_ASSOCIATIONS_EXTRACTION_TEXT_TEMPLATE="\n\nText: \"{}\""



#
# Prompt for questions generation from a given text
#

QUESTIONS_GENERATION_SYSTEM=(
    "You read a text, break it down in a sequence of factual "
    "associations sentences and generate questions about them."
)

QUESTIONS_GENERATION_PROMPT=(
    "Read the text and generate questions following the steps:"
    "\n1. Extract a list of factual associations from the text, "
    "including implicit information and temporal relations."
    
    "\n2. Create a list of questions and answers from the factual "
    "associations."
    
    "\nOnly output the JSON format, nothing else: "
    "{\"questions\":[{\"question\": \"<question-1>\", "
                     "\"answer\": \"<answer-1>\"}, ..., "
                    "{\"question\": \"<question-n>\", "
                     "\"answer\": \"<answer-n>\"}]."
)

QUESTIONS_GENERATION_TEXT_TEMPLATE="\n\nText: \"{}\""



#
# Prompt for questions generation from a given factual statement
#

QUESTIONS_GENERATION_FROM_STATEMENT_PROMPT=(
    "Generate questions from the simple factual statement. "
    "Do not repeat the question only changing few words. "
    "Only output the JSON format,  nothing else: "
    "{\"questions\":[{\"question:\": \"<question-1>\", "
                     "\"answer\": \"<answer-1>\"}, ..., "
                    "{\"question\": \"<question-n>\", "
                     "\"answer\": \"<answer-n>\"}]}"
)

QUESTIONS_GENERATION_FROM_STATEMENT_TEMPLATE=(
    "\n\nStatement: \"{}\""
)



#
# Class defining the access to Groq models.
#

class groq_access:

    def __init__(self,
                 api_key,
                 model):

        self.model = model
        self.client = Groq(api_key=api_key)
        

    def send_request(self, messages):
        
        completed_request = False

        while not completed_request:
            try:
                completion = self.client.chat.completions.create(model=self.model,
                                                                 messages=messages,
                                                                 temperature=0,
                                                                 max_tokens=2048,
                                                                 top_p=1,
                                                                 stream=True,
                                                                 stop=None)
    
                generated_text = ""
    
                for i, chunk in enumerate(completion):
                    generated_text += chunk.choices[0].delta.content or ""
    
                if generated_text == "":
                    print("\n\nQuota exceeded!!! Waiting for 30 seconds")
    
                    time.sleep(30)
                else:
                    try:
                        # Basic output cleanup
                        print("\n\n")
                        print(generated_text)
                        print("\n\n")

                        cleaned_text = generated_text.replace("\n", "")
                        cleaned_text = cleaned_text[:cleaned_text.rfind("}") + 1]

                        # print("\n\n")
                        # print(cleaned_text)
                        # print("\n\n")
                        
                        response = json.loads(cleaned_text)
                    except Exception as e:
                        print(e)
                        print("\nError while parsing the response to JSON={}\n".format(generated_text)) 
                    
                    response['generated_text'] = generated_text
                    response['prompt_tokens'] = chunk.x_groq.usage.prompt_tokens
                    response['completion_tokens'] = chunk.x_groq.usage.completion_tokens
                    response['total_tokens'] = chunk.x_groq.usage.total_tokens
                    response['total_time'] = chunk.x_groq.usage.total_time
    
                    completed_request = True
                    
            except Exception as e:
                print(e)
                print("\nError while interacting with Groq API\n")

                time.sleep(10)

        return response



#
# Function to format a message into chat format, according to the
# given role.
#

def format_message(which_role: str, which_message: str):
    return {"role": which_role,
            "content": which_message}



#
# Function to execute factual association extraction from a given text
#

def factual_association_extraction(LLM_access: groq_access, 
                                   which_text: str, 
                                   verbose=True):
    
    messages = [format_message("system", FACTUAL_ASSOCIATIONS_EXTRACTION_SYSTEM)]

    user_message = FACTUAL_ASSOCIATIONS_EXTRACTION_PROMPT + \
                   FACTUAL_ASSOCIATIONS_EXTRACTION_TEXT_TEMPLATE.format(which_text)
    
    if verbose:
        print("\n{}".format(user_message))

    messages.append(format_message("user", user_message))
    
    print(messages)

    result = LLM_access.send_request(messages)

    if verbose:
        print("\n{}".format(result))
    
    return result



#
# Function to execute simple factual association extraction from a given text
#

def simple_factual_association_extraction(LLM_access: groq_access, 
                                          which_text: str, 
                                          verbose=True):
    
    messages = [format_message("system", SIMPLE_FACTUAL_ASSOCIATIONS_EXTRACTION_SYSTEM)]

    user_message = SIMPLE_FACTUAL_ASSOCIATIONS_EXTRACTION_PROMPT + \
                   SIMPLE_FACTUAL_ASSOCIATIONS_EXTRACTION_TEXT_TEMPLATE.format(which_text)
    
    if verbose:
        print("\n{}".format(user_message))

    messages.append(format_message("user", user_message))
    
    print(messages)

    result = LLM_access.send_request(messages)

    if verbose:
        print("\n{}".format(result))
    
    return result



#
# Function to execute questions generation from a given text
#

def questions_generation(LLM_access: groq_access, 
                         which_text: str, 
                         verbose=True):
    
    messages = [format_message("system", QUESTIONS_GENERATION_SYSTEM)]

    user_message = QUESTIONS_GENERATION_PROMPT + \
                   QUESTIONS_GENERATION_TEXT_TEMPLATE.format(which_text)
    
    if verbose:
        print("\n{}".format(user_message))

    messages.append(format_message("user", user_message))
    
    print(messages)

    result = LLM_access.send_request(messages)

    if verbose:
        print("\n{}".format(result))
    
    return result



#
# Function to execute questions generation from a given statement
#

def questions_generation_from_statement(LLM_access: groq_access, 
                                        which_statement: str, 
                                        verbose=True):
    
    user_message = QUESTIONS_GENERATION_FROM_STATEMENT_PROMPT + \
                   QUESTIONS_GENERATION_FROM_STATEMENT_TEMPLATE.format(which_statement)
    
    if verbose:
        print("\n{}".format(user_message))

    result = LLM_access.send_request([format_message("user", user_message)])

    if verbose:
        print("\n{}".format(result))
    
    return result