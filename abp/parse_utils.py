def clean_tuple_str(tuple_str):
    tuple_str = tuple_str
    tuple_str_1 = tuple_str.strip().split(' ')[0]
    tuple_str_2 = tuple_str.strip().split(' ')[2]
    tuple_str = tuple_str.strip()
    return tuple_str_1, tuple_str_2, tuple_str


def parse_reasoning_output(output_str) -> dict:
    """Parse dependency gen result string into dict"""

    if 'output:' in output_str:
        start_index = output_str.index('output:')
        output_str = output_str[start_index+len('output:'):]
        output_str = output_str.strip()
        # print('refined: ', output_str)

    id2reasoning = {}
    for id_reasoning in output_str.strip().split('\n'):
        reasoning_id, reasoning, category, subcategory, question, choices, answer = id_reasoning.split('|')

        reasoning_id = reasoning_id.strip()
        reasoning = reasoning.strip()
        reasoning_id = int(reasoning_id)
        category = category.strip()
        subcategory = subcategory.strip()
        question = question.strip()
        choices = choices.strip()
        answer = answer.strip()

        id2reasoning[reasoning_id] = reasoning, category, subcategory, question, choices, answer

    return id2reasoning


def parse_tuple_output(output_str) -> dict:
    """Parse dependency gen result string into dict"""

    id2tup = {}
    for id_tup in output_str.strip().split('\n'):
        tup_id, tup, question = id_tup.split('|')
        tup_id = tup_id.strip()
        tup = tup.strip()
        question = question.strip()
        
        tuple_str_1, tuple_str_2, tuple_str = clean_tuple_str(tup)

        tup_id = int(tup_id)

        id2tup[tup_id] = tuple_str_1, tuple_str_2, tuple_str, question

    return id2tup





