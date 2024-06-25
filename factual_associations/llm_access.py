from groq import Groq

import time
import json


GROQ_LLAMA3_70B_MODEL="llama3-70b-8192"
GROQ_LLAMA3_8B_MODEL="llama3-8b-8192"


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
    "the implicit and temporal information. For each factual association, "
    "identify the subject, the relation and the object. Break down the information "
    "in sentences containing a simple object; do not create sentences with "
    "long objects. Only output the JSON format, nothing else before or after: "
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
    "Do not create a generic question. "
    "Only output the JSON format, nothing else: "
    "{\"questions\":[{\"question:\": \"<question-1>\", "
                     "\"answer\": \"<answer-1>\"}, ..., "
                    "{\"question\": \"<question-n>\", "
                     "\"answer\": \"<answer-n>\"}]}"
)

QUESTIONS_GENERATION_FROM_STATEMENT_TEMPLATE=(
    "\n\nStatement: \"{}\""
)



#
# Prompt for answers evaluation
#

ANSWERS_EVALUATION_SYSTEM=(
    "You evaluate a list of answers, taking a (question, answer) "
    "pair as reference."
)

ANSWERS_EVALUATION_PROMPT=(
    "Provide a score from 0 to 3 for the list of candidate answers, "
    "considering a pair of (reference_question, reference_answer), "
    "according to the following procedure:"
    
    "\n1. Start with score 3;"

    "\n2. If the candidate answer does not include any information "
         "in the reference_answer, attribute score 0."

    "\n3. If the candidate answer only partially includes the "
         "reference_answer information, decrement 1 point;"
    
    "\n4. If the candidate answer includes information not present "
         "in the reference question, decrement 1 point;"
    
    "\n5. If the candidate answer end in an incomplete sentence, "
         "decrement 1 point;"
    
    "\n6. If the candidate answer refers to a different entity "
         "from reference_question, attribute score 0."
    
    "\n\nProvide your answer only in JSON, nothing else: "
    "{\"reason\":\"<your-reasoning-for-the-score>\", "
     "\"score\":\"<answer-score>\"}."
)

ANSWERS_EVALUATION_TEMPLATE=(
    "\n\nreference_question: \"{}\""
        "reference_answer: \"{}\""
        "\ncandidate answer: \"{}\""
)



#
# Prompt for RAG
#

RAG_SYSTEM=(
    "You read passages and answer to a question based only on "
    "the information you got from the passages."
)

RAG_ZERO_SHOT_PROMPT=(
    "Read the passages below and answer to the question, considering "
    "only the information from the passages. Indicate if there is no "
    "answer, answering \"Not enough information\". Provide your answer "
    "in JSON, nothing else: "
    "{\"reasoning\": \"<which-information-in-which-passage-led-to-your-answer>\", "
     "\"answer\": \"<your-answer>\"}"
)

RAG_PASSAGE_TEMPLATE=(
    "\nPassage {}: \"{}\""
)

RAG_QUESTION_TEMPLATE="\nQuestion: \"{}\""



#
# 3-step factual association extraction
#

FACTUAL_ASSOCIATIONS_3_STEP_EXTRACT_SYSTEM=(
    "You read a text and break it down in a sequence of factual "
    "associations sentences."    
)

FACTUAL_ASSOCIATIONS_3_STEP_EXTRACT_PROMPT=(
    "Read the text and return a list of all factual associations you can "
    "extract only from the text information. Write independent and complete "
    "sentences; repeat the main subject to avoid pronouns.\n\nOnly output the "
    "JSON format, nothing else: {\"comments\": \"<any-comment>\", "
                                "\"sentences\": [\"<sentence-1>\", ..., "
                                                "\"<sentence-n>\"]}"
)

FACTUAL_ASSOCIATIONS_3_STEP_EXTRACT_TEMPLATE=(
    "\n\nText: \"{}\""
)

FACTUAL_ASSOCIATIONS_3_STEP_SPLIT_PROMPT=(
    "Break each sentence in \"subject\", \"relation\" and \"object\":"
    "\n1. Identify the \"subject\" of the relation;"
    "\n2. Identify the minimal \"object\" of the relation, including up to "
    "the last 3 words of the original sentence;"
    "\n3. Include in the \"relation\" every word between the \"subject\" "
    "and \"object\".\n4. Don't create a \"relation\" with only verb."

    "\n\nOnly output the JSON format, nothing else before or after: "
    "{\"sentences\":[{\"subject\":\"<subject-1>\", "
                     "\"relation\":\"<relation-1>\", "
                     "\"object\":\"<object-1>\"}, ..., "
                    "{\"subject\":\"<subject-n>\", "
                     "\"relation\":\"<relation-n>\", "
                     "\"object\":\"<object-n>\"}]}"
    "\n\nSentences:\n" 
)

FACTUAL_ASSOCIATIONS_3_STEP_SPLIT_TEMPLATE=(
    "\"{}\"\n"
)

FACTUAL_ASSOCIATIONS_3_STEP_REWRITE_PROMPT_TEMPLATE=(
    "Rewrite the sentence keeping the exact same meaning, without changing "
    "the \"subject\".\nOnly output the JSON format, nothing else before or "
    "after: {{\"sentence\": {{\"subject\":\"<new-subject>\", "
                             "\"relation\":\"<new-relation>\", "
                             "\"object\":\"<new-object>\"}}"
    "\n\nSentence:\n{}" 
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
        

    def send_request(self, messages, temperature=0):
        
        completed_request = False

        while not completed_request:
            try:
                completion = self.client.chat.completions.create(model=self.model,
                                                                 messages=messages,
                                                                 temperature=temperature,
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
                        print("\n\n---------------------")
                        print(generated_text)
                        print("---------------------\n\n")

                        cleaned_text = generated_text.replace("\n", "")

                        if cleaned_text.rfind("}") > 0:
                            cleaned_text = cleaned_text[:cleaned_text.rfind("}") + 1]
                        else:
                            cleaned_text += "}"

                        print("\n\n---------------------")
                        print(cleaned_text)
                        print("---------------------\n\n")
                        
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



#
# Function to execute answer evaluation
#

def answer_evaluation(LLM_access: groq_access, 
                      which_reference: str, 
                      which_candidate: str,
                      verbose=True):
    
    messages = [format_message("system", ANSWERS_EVALUATION_SYSTEM)]

    user_message = ANSWERS_EVALUATION_PROMPT + \
                   ANSWERS_EVALUATION_TEMPLATE.format(which_reference['question'],
                                                      which_reference['answer'], 
                                                      which_candidate)
    
    if verbose:
        print("\n{}".format(user_message))

    messages.append(format_message("user", user_message))
    
    print(messages)

    result = LLM_access.send_request(messages)

    if verbose:
        print("\n{}".format(result))
    
    return result



#
# Function to execute RAG
#

def execute_RAG(LLM_access: groq_access, 
                which_reference_passages, 
                which_question: str,
                temperature=0,
                verbose=True):
    
    system_role = [format_message("system", RAG_SYSTEM)]

    reference_passages_message = ""

    if (which_reference_passages is not None) and (len(which_reference_passages) > 0):
        for i, passage_text in enumerate(which_reference_passages):
            reference_passages_message += RAG_PASSAGE_TEMPLATE.format(i, passage_text)

    user_message = RAG_ZERO_SHOT_PROMPT + \
                   reference_passages_message + \
                   RAG_QUESTION_TEMPLATE.format(which_question)

    if verbose:
        print(user_message)

    evaluation = LLM_access.send_request(system_role + [format_message("user", user_message)],
                                         temperature=temperature)

    evaluation['question'] = which_question

    if verbose:
        print("{}\n".format(evaluation))

    return evaluation



#
# 3-step factual association extraction 
#

def factual_association_3_step_extraction(LLM_access: groq_access, 
                                          which_text: str, 
                                          verbose=True):

    # Extract the factual association complete sentences

    messages = [format_message("system", FACTUAL_ASSOCIATIONS_3_STEP_EXTRACT_SYSTEM)]

    user_message = FACTUAL_ASSOCIATIONS_3_STEP_EXTRACT_PROMPT + \
                   FACTUAL_ASSOCIATIONS_3_STEP_EXTRACT_TEMPLATE.format(which_text)
    
    if verbose:
        print("\n{}".format(user_message))

    messages.append(format_message("user", user_message))
    
    if verbose:
        print(messages)

    extraction_result = LLM_access.send_request(messages)

    if verbose:
        print("\n{}".format(extraction_result))


    # Break the sentences in "subject", "relation", and "object".

    extracted_sentences = ""

    for sentence in extraction_result['sentences']:
        extracted_sentences += FACTUAL_ASSOCIATIONS_3_STEP_SPLIT_TEMPLATE.format(sentence)

    user_message = FACTUAL_ASSOCIATIONS_3_STEP_SPLIT_PROMPT + extracted_sentences

    if verbose:
        print("\n{}".format(user_message))    

    split_result = LLM_access.send_request([format_message("user", user_message)])

    if verbose:
        print("\n{}".format(split_result))


    # Check for repeated relations

    extracted_relations = []
    verified_sentences = []
    rewrite_results = []

    for sentence in split_result['sentences']:
        if sentence['relation'].lower() in extracted_relations:

            if verbose:
                print("Repeated relation: \"{}\"\n{}".format(sentence['relation'].lower(), 
                                                             sentence))

            user_message = [format_message("user", 
                                           FACTUAL_ASSOCIATIONS_3_STEP_REWRITE_PROMPT_TEMPLATE.format(sentence))]

            rewrite_result = LLM_access.send_request(user_message)
            verified_sentences.append(rewrite_result['sentence'])

            rewrite_results.append(rewrite_result)
        else:
            extracted_relations.append(sentence['relation'].lower())
            verified_sentences.append(sentence)

    return {"sentences": verified_sentences,
            "extraction_result": extraction_result,
            "split_result": split_result,
            "rewrite_results": rewrite_results}