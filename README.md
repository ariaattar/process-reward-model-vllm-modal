# Preference Reward Model (PRM) API
A FastAPI service for evaluating LLM responses using Skywork's Preference Reward Models (PRM).

---

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
  "problems": ["User input"],
  "responses": ["Model response step 1. \n Model response step 2. \n Model response step 3. \n Model response step 4. \n Model response step 5."]
}' \
https://your-modal-endpoint.run/score
```
---

## Overview
A FastAPI service that evaluates LLM-generated responses using the Skywork-o1-Open-PRM-Qwen models. The PRM scores responses step-by-step, providing quality metrics for each step. Deployed on Modal with vLLM.

These models are used during training to improve reasoning capabilities of large language models. They help evaluate and reward high-quality reasoning steps, similar to the approach used in training models like OpenAI's o1. The PRM acts as a reward function that guides the model toward producing better structured, more logical reasoning patterns.

### Models Available
- **Default**: [Skywork-o1-Open-PRM-Qwen-2.5-1.5B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B)
- **Alternative**: [Skywork-o1-Open-PRM-Qwen-2.5-7B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B)

### How It Works
1. Breaks down responses into steps (delimited by ''\n'')
2. Evaluates each step individually
3. Returns reward scores (0-1) per step

### Use Cases
- Quality assessment of LLM outputs: Evaluate coherence, reasoning quality, and validate outputs against quality thresholds
- Training feedback for reinforcement learning: Generate reward signals to guide models toward better reasoning patterns
- Best-of-N sampling for response selection: Score and select highest-quality responses from multiple generated candidates
- Action-level reinforcement learning in agents: Enable fine-grained optimization by scoring and reinforcing individual reasoning steps

### API Response Format
```python
{
  "step_rewards": [
    [0.78, 0.70, 0.64, 0.62],  # Scores for each step
    [0.80, 0.78, 0.82, 0.83]   # Multiple responses supported
  ]
}
```

### Deployment Configuration
- GPU: H100
- Timeout: 6 minutes
- Container idle timeout: 2 minutes
- Concurrent inputs: 200
- Persistent volume for model weights

---

## Citation
```bibtex
@misc{skyworkopeno12024,
  title={Skywork-o1 Open Series},
  author={Skywork-o1 Team},
  year={2024},
  howpublished={\url{https://huggingface.co/Skywork}}
}
```

---

## References
- [Skywork-o1-PRM Inference GitHub Repository](https://github.com/SkyworkAI/skywork-o1-prm-inference/tree/main)
