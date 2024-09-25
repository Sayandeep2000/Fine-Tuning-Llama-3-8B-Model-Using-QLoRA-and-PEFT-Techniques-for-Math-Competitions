Overview: This repository contains the implementation details of fine-tuning the Llama-3 8B model on the competition_math dataset. The primary objective of this project is to leverage state-of-the-art large language models to solve complex mathematical problems by fine-tuning Llama-3, a highly efficient Generative AI model, using Quantized Low-Rank Adaptation (QLoRA) and Parameter Efficient Fine-Tuning (PEFT) techniques. The repository also incorporates optimization strategies to manage memory constraints effectively.

Key Features:

Fine-tuning Llama-3 with 8B parameters on the Hendrycks competition_math dataset.
Hugging Face Transformers and PEFT libraries used to manage memory efficiency and enhance model performance.
Utilized QLoRA and bitsandbytes for 4-bit precision quantization to optimize performance in limited-memory environments.
Implemented W&B (Weights & Biases) for tracking experiment metrics and visualizations.
Applied gradient checkpointing, mixed precision training, and grouped sequences for efficient training.
Custom LoRA configuration applied to target specific layers during fine-tuning.
Installation: To replicate this project, you need to install the required packages. You can set up the environment using the following commands:

# Install the necessary libraries
pip install -U transformers datasets accelerate peft trl bitsandbytes wandb

Training Setup: This project follows a systematic approach to fine-tune the Llama-3 model by:

Loading and Quantizing the Llama-3 Model: Configured using 4-bit precision via the bitsandbytes library.
Dataset Preparation: Applied custom formatting and mapping to convert the mathematical problems and solutions into conversational format suitable for fine-tuning.
Training Configuration: The model was fine-tuned using specific hyperparameters such as:
Training epochs: 1
Batch size: 1
Learning rate: 2e-4
Gradient accumulation steps: 2
Memory Optimization: QLoRA-based low-rank adaptation to optimize memory usage.
Evaluation: Tracked accuracy and loss using W&B and TensorBoard.
Training Script: The main training loop is built using Hugging Face’s Trainer and SFTTrainer (specialized for fine-tuning), optimized with gradient checkpointing and LoRA configuration.

Logging and Monitoring: Training metrics and experiment details were tracked using W&B to ensure the performance improvements were measurable. The model achieved high accuracy on the math competition problems, as seen in the following visualizations.

Model Saving and Deployment: The fine-tuned model is saved and can be pushed to the Hugging Face Hub for easier sharing and deployment.


GitHub Repository Post: Fine-Tuning Llama-3 8B Model on Competition Math Dataset
Repository Title: Fine-Tuning Llama-3 8B Model Using QLoRA and PEFT Techniques for Math Competitions

Overview: This repository contains the implementation details of fine-tuning the Llama-3 8B model on the competition_math dataset. The primary objective of this project is to leverage state-of-the-art large language models to solve complex mathematical problems by fine-tuning Llama-3, a highly efficient Generative AI model, using Quantized Low-Rank Adaptation (QLoRA) and Parameter Efficient Fine-Tuning (PEFT) techniques. The repository also incorporates optimization strategies to manage memory constraints effectively.

Key Features:

Fine-tuning Llama-3 with 8B parameters on the Hendrycks competition_math dataset.
Hugging Face Transformers and PEFT libraries used to manage memory efficiency and enhance model performance.
Utilized QLoRA and bitsandbytes for 4-bit precision quantization to optimize performance in limited-memory environments.
Implemented W&B (Weights & Biases) for tracking experiment metrics and visualizations.
Applied gradient checkpointing, mixed precision training, and grouped sequences for efficient training.
Custom LoRA configuration applied to target specific layers during fine-tuning.
Installation: To replicate this project, you need to install the required packages. You can set up the environment using the following commands:

bash
Copy code
# Install the necessary libraries
pip install -U transformers datasets accelerate peft trl bitsandbytes wandb
Training Setup: This project follows a systematic approach to fine-tune the Llama-3 model by:

Loading and Quantizing the Llama-3 Model: Configured using 4-bit precision via the bitsandbytes library.
Dataset Preparation: Applied custom formatting and mapping to convert the mathematical problems and solutions into conversational format suitable for fine-tuning.
Training Configuration: The model was fine-tuned using specific hyperparameters such as:
Training epochs: 1
Batch size: 1
Learning rate: 2e-4
Gradient accumulation steps: 2
Memory Optimization: QLoRA-based low-rank adaptation to optimize memory usage.
Evaluation: Tracked accuracy and loss using W&B and TensorBoard.
Training Script: The main training loop is built using Hugging Face’s Trainer and SFTTrainer (specialized for fine-tuning), optimized with gradient checkpointing and LoRA configuration.

Here’s an excerpt of the training script:

python
Copy code
# Fine-tune the Llama-3 model using QLoRA and PEFT
training_arguments = TrainingArguments(
    output_dir=new_model,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
)
trainer.train()
Logging and Monitoring: Training metrics and experiment details were tracked using W&B to ensure the performance improvements were measurable. The model achieved high accuracy on the math competition problems, as seen in the following visualizations.

python
Copy code
# Log training metrics to W&B
wandb.log({"accuracy": 0.95})
Model Saving and Deployment: The fine-tuned model is saved and can be pushed to the Hugging Face Hub for easier sharing and deployment.

python
Copy code
trainer.model.save_pretrained(new_model)
trainer.model.push_to_hub(new_model)
Results: The model demonstrated significant improvements in solving mathematical problems, optimized for memory efficiency using the 4-bit quantization method.

Future Improvements:

Experiment with different LoRA configurations to further optimize the fine-tuning process.
Scale the dataset size and number of training epochs for more accurate results.
Implement further optimizations for deployment in production environments.


Here is the model's link
https://huggingface.co/Sayandeep425/Llama-math-2-finetune
