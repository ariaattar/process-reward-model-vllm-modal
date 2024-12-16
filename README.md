# Preference Reward Model (PRM) API Explanation

## Overview
This code implements a FastAPI-based service that runs a Preference Reward Model (PRM) for evaluating the quality of LLM-generated responses. It uses the Skywork-o1-Open-PRM-Qwen models, available in both 1.5B and 7B parameter versions.

## Models
- **Default Model**: Skywork-o1-Open-PRM-Qwen-2.5-1.5B
- **Alternative Model**: Skywork-o1-Open-PRM-Qwen-2.5-7B
Both models are optimized for step-by-step solution evaluation and can be selected based on your performance and resource requirements.

## How It Works

### Architecture
- The service is deployed using Modal, a cloud platform for running serverless ML models
- Uses VLLM for efficient model serving
- Implements API key authentication for security
- Runs on H100 GPUs for fast inference

### The Reward Model
The PRM evaluates the quality of responses by:
1. Breaking down responses into steps (separated by '\n')
2. Scoring each step individually
3. Providing a reward score between 0 and 1 for each step

### Input
- The '\n' in the input serves as a step delimiter
- Each step is processed separately to evaluate its quality
- The model looks at both the problem and the response to provide context-aware scoring

## Purpose
The PRM serves several key purposes:
1. **Quality Assessment**: Evaluates how well an LLM solves problems step by step
2. **Step-by-Step Analysis**: Identifies which parts of a solution are strong or weak
3. **Training Feedback**: Can be used to improve LLM outputs through reinforcement learning
4. **Response Validation**: Helps ensure generated responses meet quality standards
5. **Action-Level Reinforcement Learning**: Enables fine-grained reward signals for training AI agents by evaluating individual actions and their consequences
6. **Best-of-N Sampling**: Supports agent decision-making by scoring multiple potential actions/responses and selecting the highest-rated option

## Usage
The API exposes two main endpoints:
- `/score`: Evaluates responses against problems

### Sample API Request
```bash
curl -X POST \
-H "X-API-Key: sk_llm_prod_123456789" \
-H "Content-Type: application/json" \
-d '{
  "problems": [
    "Janet'\''s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers'\'' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers'\'' market?",
    "Janet'\''s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers'\'' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers'\'' market?"
  ],
  "responses": [
    "To determine how much money Janet makes every day at the farmers'\'' market, we need to follow these steps:\n1. **Calculate the total number of eggs laid by the ducks per day.**\n   Janet'\''s ducks lay 16 eggs per day.\n2. **Determine the number of eggs Janet uses each day.**\n   - She eats 3 eggs for breakfast every morning.\n   - She bakes muffins for her friends every day with 4 eggs.\n   So, the total number of eggs used per day is:\n   \\[\n   3 + 4 = 7 \\text{ eggs}\n   \\]\n3. **Calculate the number of eggs Janet sells at the farmers'\'' market each day.**\n   Subtract the number of eggs used from the total number of eggs laid:\n   \\[\n   16 - 7 = 9 \\text{ eggs}\n   \\]\n4. **Determine how much money Janet makes from selling the eggs.**\n   She sells each egg for $2, so the total amount of money she makes is:\n   \\[\n   9 \\times 2 = 18 \\text{ dollars}\n   \\]\nTherefore, the amount of money Janet makes every day at the farmers'\'' market is \\(\\boxed{18}\\).",
    "To find out how much Janet makes every day at the farmers'\'' market, we need to follow these steps:\n1. Calculate the total number of eggs laid by the ducks: \n   - Janet'\''s ducks lay 16 eggs per day.\n2. Calculate the total number of eggs consumed by Janet and used for baking:\n   - Janet eats 3 eggs for breakfast.\n   - She bakes 4 eggs for muffins.\n   - Total eggs used: 3 (breakfast) + 4 (baking) = 7 eggs.\n3. Calculate the remaining number of eggs for sale:\n   - Total eggs laid: 16\n   - Eggs used: 7\n   - Remaining eggs: 16 - 7 = 9 eggs\n4. Calculate the total amount of money made at the farmers'\'' market:\n   - Price per egg: $2\n   - Number of eggs sold: 9\n   - Total money made: 9 * $2 = $18\nTherefore, Janet makes $\\boxed{18}$ dollars every day at the farmers'\'' market."
  ]}' \
https://your-modal-endpoint.run/score
```

### Sample Response
```json
{
  "step_rewards": [
    [
      0.7879311956428947,0.7074047009882609,0.6451199102918682,0.6297746074044134,....
    ],
    [
      0.8080672135527632,0.786622824583395,0.8244619263765707,0.8322366527818007,0.9124361604769414....
    ]
  ]
}
```

In this example:
- Each number in `step_rewards` represents the quality score (0-1) for a step in the solution
- Higher scores (closer to 1.0) indicate better quality steps
- The model evaluates each line break (`\n`) as a separate step
- The final high score (0.944994) suggests a strong conclusion to the solution

## Security
- Protected by API key authentication
- Uses environment-based configuration
- Implements proper error handling and validation 

## Citation
```bibtex
@misc{skyworkopeno12024,
  title={Skywork-o1 Open Series},
  author={Skywork-o1 Team},
  year={2024},
  month={November},
  howpublished={\url{https://huggingface.co/Skywork}},
  url={https://huggingface.co/Skywork},
}
```

## Model Source
The models are available on Hugging Face:
- [1.5B Model](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B)
- [7B Model](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B)
