import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class FinancialInference:
    def __init__(self, model_name="Ojaswa/QLoRa-Finetuned-Qwen-2.5-on-Wealth-Alpaca-Dataset"):
        """
        Initializes the model and tokenizer for inference.
        """
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for better performance
            device_map="auto",  # Automatically map to available device
            trust_remote_code=True
        )
        print("Model loaded successfully.")

    def generate_response(self, query, max_length=200):
        """
        Generates a response for a given query using the fine-tuned model.
        Args:
            query (str): User input query.
            max_length (int): Maximum length of the generated response.

        Returns:
            str: Generated response.
        """
        print(f"Generating response for query: {query}")
        prompt = f"""<|im_start|>system
You are a financial advisor providing comprehensive and detailed analysis. Provide thorough explanations with examples where appropriate.
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant"""

        # Tokenize the input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)

        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode and return the generated response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()

        return response

    def run_interactive(self):
        """
        Run an interactive session where users can input queries.
        """
        print("\nFinancial Advisory System")
        print("Type 'exit' to quit.")
        print("-" * 50)

        while True:
            query = input("\nYour question: ").strip()
            if query.lower() == "exit":
                print("Goodbye!")
                break

            try:
                response = self.generate_response(query)
                print("\n=== Response ===")
                print(response)
                print("=" * 50)
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Use the fine-tuned model for inference
    model_name = "Ojaswa/QLoRa-Finetuned-Qwen-2.5-on-Wealth-Alpaca-Dataset"
    inference = FinancialInference(model_name=model_name)
    inference.run_interactive()
