# Preference Reward Model (PRM) API

## Quick Start

### Installation
```bash
pip install modal
modal token new
modal deploy prm_reward_modal.py
```

### Sample API Usage
```bash
curl -X POST \
-H "X-API-Key: sk_llm_prod_123456789" \
-H "Content-Type: application/json" \
-d '{
  "problems": ["Your math problem here"],
  "responses": ["Step-by-step solution here"]
}' \
https://your-modal-endpoint.run/score
```

## Overview
A FastAPI service that evaluates LLM-generated responses using the Skywork-o1-Open-PRM-Qwen models. The service scores responses step-by-step, providing quality metrics for each step.

### Models Available
- **Default**: [Skywork-o1-Open-PRM-Qwen-2.5-1.5B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B)
- **Alternative**: [Skywork-o1-Open-PRM-Qwen-2.5-7B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B)

### Key Features
- Step-by-step solution evaluation
- Scores between 0-1 for each step
- H100 GPU acceleration
- API key authentication
- VLLM for efficient serving

## Technical Details

### How It Works
1. Breaks down responses into steps (delimited by '\n')
2. Evaluates each step individually
3. Returns reward scores (0-1) per step

### Use Cases
- Quality assessment of LLM outputs
- Training feedback for reinforcement learning
- Best-of-N sampling for response selection
- Action-level reinforcement learning

### API Response Format
```python
{
  "step_rewards": [
    [0.78, 0.70, 0.64, 0.62],  // Scores for each step
    [0.80, 0.78, 0.82, 0.83]   // Multiple responses supported
  ]
}
```

### Deployment Configuration
- GPU: H100
- Timeout: 6 minutes
- Container idle timeout: 2 minutes
- Concurrent inputs: 200
- Persistent volume for model weights

## Citation
```bibtex
@misc{skyworkopeno12024,
  title={Skywork-o1 Open Series},
  author={Skywork-o1 Team},
  year={2024},
  howpublished={\url{https://huggingface.co/Skywork}}
}
```
