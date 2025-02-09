# AI Alignment Testing Framework

This project implements a testing framework to study alignment and compliance behaviors in AI models, based on ["On the Reliability of Chain-of-Thought Reasoning in Large Language Models"](https://arxiv.org/pdf/2412.14093).

## Overview

The framework tests AI models' compliance behaviors under different conditions:
- With Chain of Thought (CoT) reasoning
- Without Chain of Thought reasoning
- Under monitored vs unmonitored conditions
- Using different prompt datasets (Advbench and CCP)

## Key Components

- `run.py`: Main testing script that:
  - Initializes OpenAI client (using local endpoint)
  - Loads test prompts
  - Runs compliance tests with and without CoT
  - Calculates compliance gaps
  - Generates statistics and plots

- `prompts/`:
  - `model_honly_deepseek.txt`: System prompt with CoT reasoning
  - `no_cot.txt`: System prompt without CoT reasoning
