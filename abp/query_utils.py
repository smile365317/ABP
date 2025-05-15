#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Acknowledgement: This code is based on the work from https://github.com/j-min/DSG
#

import pandas as pd
import string
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

import tqdm
from pprint import pprint


_PROMPT_TEMPLATE = string.Template("""
$preamble

$examples

$test_input_output
""".strip())

# In-context examples

_EXAMPLES_TEMPLATE = string.Template("""
$input_name: $input
$output_name: $output""".strip())

_TEST_TEMPLATE = string.Template("""
$input_name: $test_input
$output_name: """.lstrip())

# Task-specific Preambles

_TASK_NAMES = [
	"tuple",
	"reasoning",
]

_EXPLICIT_PREAMBLE = """Task: given input prompts, describe each scene with skill-specific tuples.
Do not generate same tuples again. Do not generate tuples that are not explicitly described in the prompts.
Based on the input prompts and skill-specific tuples, re-write tuple each in natural language question.
output format: id | tuple | question
""".strip()
_IMPLICIT_PREAMBLE = """Task: Based on the input prompt, analyze and identify the visual commonsense knowledge in the prompt (i.e., knowledge that can be observed visually), and categorize the commonsense knowledge. 
The primary categories include six types of scenes: Physical scenes, Chemical scenes, Animal scenes,Plant scenes, Human scenes, and Factual knowledge scenes. 
The secondary categories are based on specific commonsense reasoning derived from the context.Construct corresponding question-and-answer pairs based on the input prompts and visual commonsense reasoning.
output format: id | reasoning | category | subcategory | question | choices | answer
""".strip()

def load_explicit_data(path='tifa160-dev-anns.csv'):
	path = Path(__file__).parent / 'data' / path
	data_df = pd.read_csv(path)
	return data_df

def load_implicit_data(path='Implicit.csv'):
    path = Path(__file__).parent / 'data' / path
    data_df = pd.read_csv(path)
    return data_df


def create_train_example(
	prompt: str,
	task: str = "tuple",
	tuples: Optional[List[str]] = None,
	questions: Optional[List[str]] = None,
	choices:Optional[List[str]] = None,
    answer:Optional[List[str]] = None,
	category: Optional[List[str]] = None,
	subcategory: Optional[List[str]] = None,
	reasoning: Optional[List[str]] = None,
) -> Dict[str, str]:
	"""Create a training (shown in-context) example for tuple/dependency/question generation tasks.

	Tasks (one of _TASK_NAMES):
	tuple generation: prompt -> tuples
	dependency generation: prompt + tuples -> dependencies
	question generation: prompt + tuples -> questions

	Args:
	prompt: input text prompt
	task: one of pre-defined tasks in _TASK_NAMES
	tuples: list of semantic tuples to create evaluation queries
	dependencies: list of dependencies between evaluation queries
	questions: list of natural language queries

	Returns:
	{
		"input": str - text prompt
		"output": str - task-specific target output
	}
	"""

	assert task in _TASK_NAMES, f"task == {task}"

	inputs = []
	outputs = []
	

	if task == "tuple":
		n_outputs = len(tuples)
		inputs += [prompt]

		for i in range(n_outputs):
			output = f"{i+1} | {tuples[i]} | {questions[i]}"
			output = " ".join(output.split())  # remove double whitespaces if any
			outputs += [output]
	elif task == "reasoning":
		inputs += [prompt]
		n_outputs = len(category)
		for i in range(n_outputs):
			output = f"{i+1} | {reasoning[i]} | {category[i]} | {subcategory[i]} | {questions[i]} | {choices[i]} | {answer[i]}"
			output = " ".join(output.split())  # remove double whitespaces if any
			outputs += [output]
	return {
		"input": "\n".join(inputs),
		"output": "\n".join(outputs),
	}


def tifa_id2example(
	df: pd.DataFrame,
	id: str,
	task: str = "tuple",
) -> Dict[str, str]:
  """Create a training in-context example from TIFA annotation dataframe.

  Args:
	df: pandas dataframe with columns: [item_id, text, tuple, dependency,
	  question_natural_language]
	id: unique prompt id (item_id)
	task: one of pre-defined tasks: ["tuple", "dependency", "question"]

  Returns:
	{
		'input': str - text prompt
		'output': str - task-specific target output
	}
  """

  prompt = df[df.item_id == id].text.tolist()[0]
  all_tuples = df[df.item_id == id].tuple.tolist()
  all_questions = df[df.item_id == id].question_natural_language.tolist()

  # Create an example
  example = create_train_example(
      prompt=prompt,
      task=task,
      tuples=all_tuples,
      questions=all_questions,
  )
  return example


def implicit_id2example(
	df: pd.DataFrame,
	id: str,
	task: str = "tuple",
) -> Dict[str, str]:
  prompt = df[df.item_id == id].prompt.tolist()[0]
  all_questions = df[df.item_id == id].question_natural_language.tolist()
  all_choices = df[df.item_id == id].choices.tolist()
  all_answer = df[df.item_id == id].answer.tolist()
  all_category = df[df.item_id == id].category.tolist()
  all_subcategory = df[df.item_id == id].subcategory.tolist()
  all_reasoning = df[df.item_id == id].reasoning.tolist()

  example = create_train_example(
      prompt = prompt,
      task = task,
      questions = all_questions,
      choices = all_choices,
      answer = all_answer,
      category = all_category,
      subcategory = all_subcategory,
      reasoning = all_reasoning,
  )

  return example


def get_implicit_examples(data_df, ids, task='tuple'):
	examples = []
	for id in ids:
		example = implicit_id2example(data_df, id, task=task)
		examples += [example]
	return examples
def get_tifa_examples(data_df, ids, task='tuple'):
	examples = []
	for id in ids:
		example = tifa_id2example(data_df, id, task=task)
		examples += [example]
	return examples


TIFA160_ICL_TRAIN_IDS = [
	'coco_361740',
	'drawbench_155',
	'partiprompt_86',
	'paintskill_374',
	'coco_552592',
	'partiprompt_1414',
	'coco_627537',
	'coco_744388',
	'partiprompt_1108',
	'coco_397109',
	'coco_666114',
	'coco_62896',
	'paintskill_235',
	'drawbench_159',
	'partiprompt_893',
	'coco_322041',
	'coco_292534',
	'drawbench_57',
	'partiprompt_555',
	'coco_488166',
	'partiprompt_726',
	'coco_323167',
	'coco_625027',
]
IMPLICIT_ICL_TRAIN_IDS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]

_IMPLICIT_DF = load_implicit_data()
_IMPLICIT_EXAMPLES = get_implicit_examples(_IMPLICIT_DF, IMPLICIT_ICL_TRAIN_IDS, task='reasoning')
_TIFA160_DF = load_explicit_data()
_TUPLE_EXAMPLES = get_tifa_examples(_TIFA160_DF, TIFA160_ICL_TRAIN_IDS, task='tuple')

def make_prompt(
	examples: List[Dict[str, str]],
	test_input: str,
	preamble: str ,
	input_name: str = "input",
	output_name: str = "output",
	verbose: bool = False,
) -> str:
	"""Make a prompt by composing preamble, examples, and text input.

	Args:
	examples: list of examples - each example has keys ['input', 'output']
	test_input: test input string to generate output
	preamble: a task description for language model
	input_name: a verbalizer for input
	output_name: a verbalizer for output
	verbose: whether to print the prompt details (e.g., prompt length)

	Returns:
	prompt (str)

	Example output:

	Task: given input prompts, describe each scene with skill-specific tuples.
	Do not generate same tuples again. Do not generate tuples that are not
	explicitly described in the prompts.
	output format: id | tuple
	input: A red motorcycle parked by paint chipped doors.
	output: 0 | attribute - color (motorcycle, red)
	1 | attribute - state (door, paint chipped)
	2 | relation - spatial (motorcycle, door, next to)
	3 | attribute - state (motorcycle, parked)
	input: a large clock hangs from a building and reads 12:43.
	output: 0 | attribute - scale (clock, large)
	...
	input: A dignified beaver wearing glasses, a vest, and colorful neck tie. He stands next to a tall stack of books in a library.
	output:
	"""

	# examples: list of "input: $input \n output: $output"
	examples_str = []
	for example in examples:
		examples_str.append(
		_EXAMPLES_TEMPLATE.substitute(
			input_name=input_name,
			output_name=output_name,
			input=example["input"].strip(),
			output=example["output"].strip(),
		)
	)
	examples_str = "\n\n".join(examples_str)

	test_input_str = _TEST_TEMPLATE.substitute(
		input_name=input_name,
		output_name=output_name,
		test_input=test_input
	)

	prompt = _PROMPT_TEMPLATE.substitute(
		preamble=preamble,
		examples=examples_str,
		test_input_output=test_input_str,
	)

	if verbose:
		print(f"len(preamble): {len(preamble)}chars & {len(preamble.split())}words")
		print(f"len(examples): {len(examples)}chars & {len(examples_str)}words")
		print(f"len(total): {len(prompt)}chars & {len(prompt.split())}words")

	return prompt


def parse_with_input_name(text: str, input_name="input") -> str:
  """Parse the first LM output by splitting with input verbalizer."""
  text = text.split(f"{input_name}:")[0]
  return text


def generate_with_in_context_examples(
	generate_fn: Callable[[str], str],
	id2inputs: Dict[str, Dict[str, str]],
	train_examples: List[Dict[str, Any]],
	preamble: str,
	input_name: str = "input",
	output_name: str = "output",
	parse_fn: Callable[[str], str] = parse_with_input_name,
	num_workers: int = 1,
	verbose=True,
) -> Dict[str, Dict[str, str]]:
	"""Generate output with a language model with in-context examples.

	Args:
	generate_fn: a method that calls language model with a text input
	id2inputs: a input dictionary with following structure "id" (str) -> {
		"input": "test input prompt" (str) }
	train_examples: list of examples. Each example is a dict('input', 'output')
	preamble: a task description for language model
	input_name: a verbalizer for input
	output_name: a verbalizer for output
	parse_fn: a method that parses the output of language model.
	num_workers: number of workers for parallel call
	verbose: whether to print tqdm output / intermediate steps

	Returns:
	id2outputs: output dictionary with key with following structure
		"id" (str) -> {
		"input": "text prompt" (str),
		"output": "generated output" (str)
		}
	"""

	ids = list(id2inputs.keys())

	# 1) Create list of LM inputs
	total_kwargs = []

	for id_ in tqdm.tqdm(
		ids,
		dynamic_ncols=True,
		ncols=80,
		disable=not verbose,
		desc="Preparing LM inputs",
	):
		test_input = id2inputs[id_]["input"]

		prompt = make_prompt(
			examples=train_examples,
			test_input=test_input,
			preamble=preamble,
			input_name=input_name,
			output_name=output_name,
			verbose=False,
		)

		total_kwargs.append({"prompt": prompt})

	# 2) Run LM calls
	if verbose:
		print(f"Running LM calls with {num_workers} workers.")
	if num_workers == 1:
		total_output = []
		for kwargs in tqdm.tqdm(total_kwargs):
			prompt = kwargs["prompt"]
			output = generate_fn(prompt)
			total_output += [output]

	else:
		from multiprocessing import Pool
		with Pool(num_workers) as p:
			total_inputs = [d['prompt'] for d in total_kwargs]
			total_output = list(
				tqdm.tqdm(p.imap(generate_fn, total_inputs), total=len(total_inputs)))

	# 3) Postprocess LM outputs
	id2outputs = {}

	for i, id_ in enumerate(
		tqdm.tqdm(
				ids,
				dynamic_ncols=True,
				ncols=80,
				disable=not verbose,
				desc="Postprocessing LM outputs"
			)
		):

		test_input = id2inputs[id_]["input"]
		raw_prediction = total_output[i]
		prediction = parse_fn(raw_prediction).strip()

		out_datum = {}
		out_datum["id"] = id_
		out_datum["input"] = test_input
		out_datum["output"] = prediction

		id2outputs[id_] = out_datum

	return id2outputs


def generate_dsg(id2prompts: Dict[str, Dict[str, str]],
				 generate_fn: Callable[[str], str],
                 tuple_train_examples=_TUPLE_EXAMPLES,
                 N_parallel_workers=1,
				 verbose=False
				 ):
	eval_data = []
	for id, input_dict in id2prompts.items():
		datum = {
			'id': id,
			'prompt': input_dict['input']
		}
		eval_data.append(datum)

	test_ids = [datum['id'] for datum in eval_data]

	# =====================================
	# Task 1: Tuple generation
	# =====================================
	task, preamble = ['tuple', _EXPLICIT_PREAMBLE]

	if verbose:
		print('Task 1: ', task)

	train_examples = tuple_train_examples

	id2inputs = {}
	for i, datum in enumerate(eval_data):
		input_dict = {}

		test_prompt = datum['prompt']
		id = datum['id']

		input_dict['input'] = test_prompt

		id2inputs[id] = input_dict

	if verbose:
		print('Run inference')
	# used as inputs to task 2 (question gen) & task 3 (dependency gen)
	id2tuple_outputs = generate_with_in_context_examples(
		generate_fn=generate_fn,
		id2inputs=id2inputs,
		train_examples=train_examples,
		preamble=preamble,
		num_workers=N_parallel_workers,
		verbose=verbose)

	if verbose:
		print('Sample results:')
		for id in test_ids[:1]:
			print('id:', id)
			pprint(id2tuple_outputs[id])



	return id2tuple_outputs


def generate_dsg_implicit(id2prompts: Dict[str, Dict[str, str]],
				 generate_fn: Callable[[str], str],
                 N_parallel_workers=1,
				 verbose=False
				 ):

	eval_data = []
	for id, input_dict in id2prompts.items():
		datum = {
			'id': id,
			'prompt': input_dict['input']
		}
		eval_data.append(datum)

	test_ids = [datum['id'] for datum in eval_data]
 
	# =====================================
	# # Task : reasoning generation
	# =====================================
	task, preamble = ['reasoning', _IMPLICIT_PREAMBLE]

	if verbose:
		print('Task 1: ', task)

	train_examples = _IMPLICIT_EXAMPLES

	id2inputs = {}
	for i, datum in enumerate(eval_data):
		input_dict = {}

		test_prompt = datum['prompt']
		id = datum['id']

		input_dict['input'] = test_prompt

		id2inputs[id] = input_dict

	if verbose:
		print('Run inference')
	# used as inputs to reasoning question
	id2reasoning_outputs = generate_with_in_context_examples(
		generate_fn=generate_fn,
		id2inputs=id2inputs,
		train_examples=train_examples,
		preamble=preamble,
		num_workers=N_parallel_workers,
		verbose=verbose)

	if verbose:
		print('Sample results:')
		for id in test_ids[:1]:
			print('id:', id)
			pprint(id2reasoning_outputs[id])
	
 
	return id2reasoning_outputs