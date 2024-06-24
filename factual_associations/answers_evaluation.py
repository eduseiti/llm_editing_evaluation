import pandas as pd
import numpy as np
import time

from llm_access import *



#
# Configure Pandas Dataframes output
#

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 500)

#
# Function to compute the scores for a sequence of questions
#

def evaluate_questions(groq_interface,
                       which_questions,
                       edit_round_number=0):

    start_time = time.time()

    evaluations = {}
    evaluations['round'] = edit_round_number
    evaluations['questions'] = []
    
    for question in which_questions:

        print("\n>> Question: {}".format(question['question']['question']))

        question_result = {}
        
        question_result['question'] = question['question']['question']

        question_scores = []
        question_evaluations = []

        for answer in question['answers']:
            score = answer_evaluation(groq_interface, 
                                      question['question'],
                                      answer)

            question_scores.append(int(score['score']))

            score['candidate_answer'] = answer
            
            question_evaluations.append(score)

        print(question_scores)
        
        question_result['mean_score'] = np.mean(question_scores)
        question_result['std_score'] = np.std(question_scores)
        question_result['evaluations'] = question_evaluations

        evaluations['questions'].append(question_result)

    evaluations['total_time'] = time.time() - start_time

    return evaluations


#
# Function to compute the scores of all statements sent up to a given edit round
#

def evaluate_statement_questions(groq_interface,
                                 statements_questions,
                                 statements_scores,
                                 edit_round_number=0):

    start_time = time.time()
    
    for statement in statements_questions:

        statement_start_time = time.time()
        
        print("\nStatement: {}".format(statement['statement']))
        
        if statement['statement'] not in statements_scores:
            statements_scores[statement['statement']] = []

        statement_round = evaluate_questions(groq_interface,
                                             statement['answers'],
                                             edit_round_number=edit_round_number)
        
        statements_scores[statement['statement']].append(statement_round)

    end_time = time.time()

    return end_time - start_time



#
# Function to group the evaluation results in a list for table creation
#

def format_evaluation_results(which_evaluation, 
                              statement=None,
                              results_table=None, 
                              generate_table=False):

    if results_table is None:
        results_table = []

    for question in which_evaluation['questions']:
        results = {}

        if statement is not None:
            results['statement'] = statement

        results['round'] = which_evaluation['round']
        results['question'] = question['question']
        results['mean_score'] = question['mean_score']
        results['std_score'] = question['std_score']

        results_table.append(results)

    if generate_table:
        return pd.DataFrame(results_table)
    else:
        return results_table



#
# Function to create a table from the statements answers evaluation for all edit rounds
#

def create_evaluation_table(statements_scores):

    results_table = []
    
    for statement, rounds in statements_scores.items():
    
        print(statement)
        
        for evaluation in rounds:
            format_evaluation_results(evaluation,
                                      statement=statement,
                                      results_table=results_table)
    
    return pd.DataFrame(results_table)



