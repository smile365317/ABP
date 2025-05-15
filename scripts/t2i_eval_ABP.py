
import json
import pandas as pd
import base64
import io
import ast
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from abp.openai_utils import openai_completion
from abp.query_utils import generate_dsg, generate_dsg_implicit
from abp.parse_utils import parse_tuple_output, parse_reasoning_output
import openai

openai.api_key = ""

# ABP Processor for processing the chunk data
class ABPProcessor:
    def __init__(self, config=None):
        self.config = config or {}
        self.combined_results = []
    
    def load_json_data(self, file_path):
        """Read JSON data from file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    def save_final_results(self, output_file):
        """Save final results to CSV"""
        if not self.combined_results:
            print("No results to save")
            return
        
        df = pd.DataFrame(self.combined_results)
        final_df = df[['item_id', 'prompt', 'question_natural_language', 'choices', 'answer']]
        final_df.to_csv(output_file, index=False)
        print(f"Final results saved to {output_file}")
        return output_file
    
    def process_explicit_chunk(self, chunk):
        """Process explicit hallucination chunk data"""
        id2tuple_outputs = generate_dsg(chunk, generate_fn=openai_completion)
        
        for item_id, item_data in chunk.items():
            try:
                tuple_data = parse_tuple_output(id2tuple_outputs.get(item_id, {}).get('output', ''))
                for qid, (_, _, _, question) in tuple_data.items():
                    self.combined_results.append({
                        'item_id': item_id,
                        'prompt': item_data.get("input", ""),
                        'question_natural_language': question,
                        'choices': "['yes','no']",
                        'answer': "yes"
                    })
            except Exception as e:
                print(f"Error processing explicit item_id {item_id}: {e}")
    
    def process_implicit_chunk(self, chunk):
        """Process implicit hallucination chunk data"""
        id2reasoning_outputs = generate_dsg_implicit(chunk, generate_fn=openai_completion)
        
        for item_id, item_data in chunk.items():
            try:
                qid2reasoning = parse_reasoning_output(id2reasoning_outputs[item_id]['output'])
                for qid, (_, _, _, question, choices, answer) in qid2reasoning.items():
                    self.combined_results.append({
                        'item_id': item_id,
                        'prompt': item_data["input"],
                        'question_natural_language': question,
                        'choices': choices,
                        'answer': answer
                    })
            except Exception as e:
                print(f"Error processing implicit item_id {item_id}: {e}")
    
    def process_in_chunks(self, input_file, output_file, chunk_size=20):
        """Process data in chunks and save results"""
        id2prompts = self.prepare_data(input_file)
        total_items = len(id2prompts)
        
        for start_idx in range(0, total_items, chunk_size):
            chunk = dict(list(id2prompts.items())[start_idx:start_idx + chunk_size])
            self.process_explicit_chunk(chunk)
            self.process_implicit_chunk(chunk)
            print(f"Processed batch {start_idx//chunk_size + 1} (Items {start_idx+1} to {min(start_idx+chunk_size, total_items)})")
        
        return self.save_final_results(output_file)
    
    def prepare_data(self, json_file):
        """Prepare data dictionary from JSON"""
        json_data = self.load_json_data(json_file)
        return {item["id"]: {'input': item["prompt"]} for item in json_data}
# ABP Score Evaluator to evaluate models for visual question answering
class ABPScoreEvaluator:
    def __init__(self, models):
        self.dataset = load_dataset("smileying/ABP")["train"]
        self.models = models  # List of models to evaluate

    def encode_image(self, image_input):
        """Encode image as base64"""
        if isinstance(image_input, str):
            with open(image_input, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        elif isinstance(image_input, Image.Image):
            img_byte_arr = io.BytesIO()
            image_input.save(img_byte_arr, format=image_input.format)
            return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        raise ValueError("Invalid image input type")

    def vqa(self, image, question, choices):
        """Visual Question Answering API call for specific model"""
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"Questions: {question} Choices:{choices}, answer each question by selecting an answer from the Choices(The capitalization of each answer should be consistent with that of the Choices.). Use ';' to separate each answer."},
            ],
            image=image,
        )

        return response.choices[0].message["content"]

    def evaluate(self, question_answer_path, save_path):
        """Run evaluation for all models and calculate the final score for each"""
        df = pd.read_csv(question_answer_path)
        results = []
        processed_count = 0
        model_scores = {model: {'correct': 0, 'total': 0} for model in self.models}

        for _, row in tqdm(df.iterrows(), total=len(df)):
            item_id = row["item_id"]
            try:
                # Get image data
                image_data = self.dataset[item_id - 1]["gpt4o"]  # Adjust based on the dataset's key
                base64_img = self.encode_image(image_data)
                
                # Parse question and choices
                question = row['question_natural_language']
                choices = ast.literal_eval(row['choices'])

                # Evaluate for each model
                for model in self.models:
                    answer = self.vqa(base64_img, question, choices, model)
                    if not answer:
                        print(f"[Warning] No answer for item {item_id} using model {model}")
                        continue
                    
                    # Calculate score for the model
                    correct = 1 if answer == row["answer"] else 0
                    model_scores[model]['correct'] += correct
                    model_scores[model]['total'] += 1

                    results.append({
                        'item_id': item_id,
                        'prompt': row["prompt"],
                        'model': model,
                        'question': question,
                        'score': correct
                    })
                    processed_count += 1
                
            except Exception as e:
                print(f"[Error] Processing item {item_id}: {str(e)}")
        
        # Calculate final accuracy for each model
        final_scores = {}
        for model, score_data in model_scores.items():
            accuracy = (score_data['correct'] / score_data['total']) * 100 if score_data['total'] > 0 else 0
            final_scores[model] = accuracy
            print(f"Final score for {model}: {accuracy:.2f}%")

        # Save results to CSV file
        pd.DataFrame(results).to_csv(save_path, index=False)
        print(f"Evaluation completed. Results saved to {save_path}")
        
        # Return final scores for each model
        return final_scores

# Main function to orchestrate the ABP processing and evaluation
def main():
    # Configuration parameters
    config = {
        'abp_input': './data/prompts.json',
        'abp_output': './results/final_results.csv',
        'tifa_input': './results/final_results.csv',
        'tifa_output': './results/output_all_models.csv',  # Updated output file
        'models': [
            "gpt4o", "dalle3", "midjourney-1", "midjourney-2", "midjourney-3", "midjourney-4",
            "gemini2", "cogview4", "sd3.5-L", "sd3-M", "sdxl"
        ]
    }
    
    # Step 1: ABP Processing
    print("=== Running ABP Processing ===")
    abp_processor = ABPProcessor()
    abp_output = abp_processor.process_in_chunks(
        input_file=config['abp_input'],
        output_file=config['abp_output']
    )
    
    # Step 2: ABPScore Evaluation
    print("\n=== Running ABPScore Evaluation ===")
    evaluator = ABPScoreEvaluator()
    final_scores = evaluator.evaluate(
        question_answer_path=abp_output,
        save_path=config['tifa_output']
    )

    # Optionally print final scores for all models
    print("\nFinal Scores for each model:")
    for model, score in final_scores.items():
        print(f"{model}: {score:.2f}%")

if __name__ == "__main__":
    main()