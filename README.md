# LLM Evaluations Workshop
**From Fundamentals to Production-Ready Evaluation Frameworks**

## 🎯 Overview

This hands-on workshop series takes you through a progressive journey of LLM evaluation techniques, starting from basic concepts and building up to sophisticated, production-ready evaluation frameworks like the ones used in enterprise solutions such as the AutoAuth Solution Accelerator.

### Target Audience
- **Technical professionals** with basic programming experience
- **Developers** new to AI/LLM development
- **Solution architects** exploring LLM integration patterns
- **Data scientists** looking to implement systematic LLM evaluation

### Prerequisites
- Basic Python programming knowledge
- Azure subscription with access to create resources
- Familiarity with Jupyter notebooks
- Understanding of basic software development concepts

---

## 🏗️ Workshop Structure

The workshop is organized into **3 progressive labs**, each building upon the previous one. Each lab combines theory, hands-on coding, and real-world application patterns.

| Lab | Duration | Focus Area | Key Skills |
|-----|----------|------------|------------|
| **Lab 1** | 60 min | Fundamentals | Basic evaluation concepts, dataset creation, simple evaluators |
| **Lab 2** | 60–90 min | Azure AI Foundry Evaluations | Using AI Foundry evaluation tools, batch evaluation, integration patterns |
| **Lab 3** | 60–90 min | Red Teaming & Adversarial Evaluation | Red teaming practices, automated adversarial testing, safety metrics |

---

## 📚 Lab Details

### Lab 1: LLM Evaluation Fundamentals
**Notebook**: [`lab1_evaluation_fundamentals/lab1_evaluation_fundamentals.ipynb`](lab1_evaluation_fundamentals/lab1_evaluation_fundamentals.ipynb )
**Duration**: 60 minutes

#### 🎯 Learning Objectives
- Understand why LLM evaluation is critical for production systems
- Learn core evaluation metrics (relevance, coherence, groundedness)
- Perform your first evaluation using basic evaluation tools and utilities
- Recognize the non-deterministic nature of LLM outputs

#### 📋 Lab Structure
1. **Introduction (15 min)**  
   - Why traditional software testing doesn't work for LLMs  
   - Demo: Same prompt, different outputs  
   - Business impact of poor LLM performance

2. **Core Concepts (15 min)**  
   - Quality metrics: Relevance, Coherence, Fluency, Groundedness  
   - Safety metrics: Harmful content, Bias detection  
   - Performance metrics: Latency, Token usage, Cost analysis

3. **Hands-On: Basic Evaluation (25 min)**  
   - Create your first evaluation dataset (Q&A pairs)  
   - Run simple evaluators programmatically using helper utilities in [`lab1_evaluation_fundamentals/utils/lab1_helpers.py`](lab1_evaluation_fundamentals/utils/lab1_helpers.py )  
   - Interpret evaluation results

4. **Wrap-up (5 min)**: Key takeaways and preview

#### 🔧 Technical Components
```python
# Key code patterns you'll implement
from azure.ai.evaluation import evaluate, RelevanceEvaluator

# Basic evaluation setup
evaluator = RelevanceEvaluator()
results = evaluate(data=test_data, evaluators={"relevance": evaluator})
```

#### 📈 Success Metrics
- Successfully run first evaluation
- Understand metric interpretation
- Can explain why evaluation is necessary

---

### Lab 2: Azure AI Foundry Evaluations
**Notebook**: [`lab2_aifoundry_evals/Evaluate_Azure_AI_Agent_Quality.ipynb`](lab2_aifoundry_evals/Evaluate_Azure_AI_Agent_Quality.ipynb )
**Duration**: 60–90 minutes

#### 🎯 Learning Objectives
- Integrate evaluations with Azure AI Foundry and the Azure AI evaluation SDK
- Run batch evaluations and compare models using Foundry tooling
- Build reproducible evaluation workflows that log results to Azure
- Understand Foundry-specific metrics and telemetry

#### 📋 Lab Structure
1. **Foundry Overview (10–15 min)**  
   - What Azure AI Foundry provides for evaluation and monitoring  
   - Differences between local evaluators and Foundry-managed evaluation

2. **Hands-On: Foundry Setup (15–20 min)**  
   - Configure credentials and project settings (environment variables or [`.env`](.env ))  
   - Connect to Foundry clients in [`shared_utils/azure_clients.py`](shared_utils/azure_clients.py )

3. **Hands-On: Batch & Multi-Model Evaluation (25–35 min)**  
   - Create batch evaluation jobs using dataset files and programmatic APIs  
   - Compare model variants/configurations and collect standardized metrics  
   - Persist evaluation results and telemetry to Foundry for downstream analysis

4. **Best Practices & Observability (10 min)**  
   - Logging, monitoring and cost-aware evaluation strategies

#### 🔧 Technical Components
```python
# Example patterns
from shared_utils.azure_clients import create_foundry_client
from shared_utils.evaluation_helpers import run_batch_evaluation

foundry = create_foundry_client()
results = run_batch_evaluation(foundry, dataset_path="lab2_aifoundry_evals/data/test_dataset.jsonl")
```

#### 📈 Success Metrics
- Successfully connect to Azure AI Foundry for evaluation runs
- Run batch evaluations and compare at least two model configurations
- Log and inspect evaluation telemetry in Foundry

---

### Lab 3: Red Teaming & Adversarial Evaluation
**Notebook**: [`lab3_redteaming/AI_RedTeaming.ipynb`](lab3_redteaming/AI_RedTeaming.ipynb ) (and `AI_Red_Teaming_Agent_Part2.ipynb`)
**Duration**: 60–90 minutes

#### 🎯 Learning Objectives
- Design and run red-team style, adversarial tests for generative models
- Implement automated red teaming pipelines and synthetic adversarial case generation
- Measure safety, jailbreak resistance, and robustness using structured metrics
- Combine automated red teaming with manual review and triage workflows

#### 📋 Lab Structure
1. **Red Teaming Concepts (15 min)**  
   - Threat modeling for generative systems  
   - Adversarial patterns: prompt injection, jailbreaks, content steering

2. **Hands-On: Creating Red Team Cases (20–25 min)**  
   - Generate adversarial prompts programmatically (synthetic generation + curated cases)  
   - Store red-team cases in [`lab3_redteaming/red_team_output.json`](lab3_redteaming/red_team_output.json ) and related datasets

3. **Hands-On: Automated Red Team Pipeline (20–30 min)**  
   - Run automated red-team tests against model endpoints  
   - Capture safety metrics, severity scores, and rationale logging  
   - Integrate results into evaluation dashboards and alerting

4. **Triage & Remediation (10 min)**  
   - Prioritize issues and recommend hardened prompt/response filters  
   - Create regression tests to track fixes

#### 🔧 Technical Components
```python
# Example red teaming pattern
from lab3_redteaming import red_team_runner
cases = red_team_runner.load_cases("lab3_redteaming/red_team_output.json")
results = red_team_runner.run_against_model(cases, model_config)
```

#### 📈 Success Metrics
- Produce a catalog of adversarial tests
- Automate at least one red-team run and capture safety-related metrics
- Produce remediation steps and regression checks for discovered issues

---

## 🚀 Getting Started

### 1. Environment Setup

#### Clone the Workshop Repository
```bash
git clone <workshop-repository-url>
cd llm-evaluations-workshop
```

#### Create Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

#### Configure Azure Resources
```bash
# Set up environment variables
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_AI_FOUNDRY_PROJECT_NAME="your-project"

# Or create .env file these values
```

### 2. Verify Setup
Run the setup verification notebook:
```bash
jupyter notebook shared_utils/setup_verification.ipynb
```

### 3. Start with Lab 1
```bash
jupyter notebook lab1_evaluation_fundamentals/lab1_evaluation_fundamentals.ipynb
```

---

## 📁 Repository Structure

This README reflects the repository structure used in the workshop:

```
llm-evaluations-workshop/
├── README.md
├── requirements.txt
├── temp_evaluation_data.jsonl
│
├── lab1_evaluation_fundamentals/
│   ├── lab1_evaluation_fundamentals.ipynb
│   ├── data/
│   │   ├── lab1_basic_evaluation.json
│   │   └── sample_qa_pairs.jsonl
│   └── utils/
│       └── lab1_helpers.py
│
├── lab2_aifoundry_evals/
│   ├── Evaluate_Azure_AI_Agent_Quality.ipynb
│   ├── user_functions.py
│   └── data/
│       └── foundry_test_dataset.jsonl
│
├── lab3_redteaming/
│   ├── AI_RedTeaming.ipynb
│   ├── red_team_output.json
│   └── redteam.log
│
├── shared_utils/
│   ├── azure_clients.py
│   ├── evaluation_helpers.py
│   └── data_utils.py
└── docs/
    ├── evaluation_metrics_guide.md
    ├── azure_setup_guide.md
    └── troubleshooting.md
```

---

## 🎯 Workshop Outcomes

By the end of this workshop series, you will be able to:

### Technical Skills
- ✅ Implement comprehensive LLM evaluation pipelines
- ✅ Create custom evaluators for domain-specific requirements
- ✅ Integrate evaluations with Azure AI Foundry for production monitoring
- ✅ Run automated red-team tests and build remediation workflows

### Strategic Understanding
- ✅ Evaluate the quality vs. cost trade-offs of different LLM approaches
- ✅ Design evaluation strategies that scale with your application
- ✅ Implement continuous evaluation in production environments
- ✅ Build evaluation frameworks that support regulatory compliance

### Real-World Application
- ✅ Adapt AutoAuth evaluation patterns to other domains
- ✅ Integrate evaluation into CI/CD pipelines
- ✅ Create evaluation dashboards and monitoring systems
- ✅ Establish evaluation best practices for your team

---

## 📖 Additional Resources

### Documentation & References
- [Azure AI Evaluation SDK Documentation](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-evaluation-readme)
- [AutoAuth Solution Accelerator](https://github.com/Azure-Samples/autoauth-solution-accelerator)
- [Azure AI Foundry Evaluation Guide](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/evaluate-generative-ai-app)
- [LLM Evaluation Best Practices](https://learn.microsoft.com/en-us/azure/ai-studio/concepts/evaluation-approach-gen-ai)

### Community & Support
- **Workshop Q&A**: Use GitHub Issues in this repository
- **Azure AI Community**: [Microsoft Tech Community](https://techcommunity.microsoft.com/t5/azure-ai-services/ct-p/AzureAIServices)
- **Evaluation Patterns**: Check out the `lab2_aifoundry_evals/examples/` and [`lab3_redteaming`](lab3_redteaming ) for additional use cases

### Next Steps
After completing this workshop, consider exploring:
- **Advanced RAG Evaluation**: Specialized patterns for retrieval-augmented generation
- **Multi-modal Evaluation**: Evaluating vision and audio capabilities
- **A/B Testing for LLMs**: Statistical approaches to model comparison
- **Production Monitoring**: Real-time evaluation and alerting systems

---

## 🤝 Contributing

We welcome contributions to improve this workshop! Please see our Contributing Guidelines for details on:
- Reporting issues or bugs
- Suggesting new lab exercises
- Adding evaluation patterns
- Improving documentation

---

## 📄 License

This workshop is licensed under the MIT License. See LICENSE for details.

---

## 🆘 Need Help?

- **Technical Issues**: Open an issue in this repository
- **Azure Setup Problems**: Check our Azure Setup Guide
- **Evaluation Questions**: Review the Troubleshooting Guide
- **Advanced Topics**: Explore Advanced Topics Documentation