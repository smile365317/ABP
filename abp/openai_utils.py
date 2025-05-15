
import openai

def openai_completion(
	prompt,
 	model='gpt-4o',
	temperature=0,
	return_response=False,
	max_tokens=500,
	):

	resp = openai.chat.completions.create(
		model=model,
		messages=[{"role": "user", "content": prompt}],
		temperature=temperature,
		max_tokens=max_tokens,
	)
	
	if return_response:
		return resp

	return resp.choices[0].message.content