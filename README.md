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

The workshop is organized into **4 progressive labs**, each building upon the previous one. Each lab combines theory, hands-on coding, and real-world application patterns.

| Lab | Duration | Focus Area | Key Skills |
|-----|----------|------------|------------|
| **Lab 1** | 60 min | Fundamentals | Basic evaluation concepts, AI Foundry basics |
| **Lab 2** | 60 min | Scaling | Dataset creation, batch evaluation, model comparison |
| **Lab 3** | 60 min | Customization | Custom evaluators, domain-specific metrics |
| **Lab 4** | 60 min | Enterprise Patterns | Configuration-driven frameworks, production deployment |

---

## 📚 Lab Details

### Lab 1: LLM Evaluation Fundamentals
**Notebook**: `lab1_evaluation_fundamentals.ipynb`
**Duration**: 60 minutes

#### 🎯 Learning Objectives
- Understand why LLM evaluation is critical for production systems
- Learn core evaluation metrics (relevance, coherence, groundedness)
- Perform your first evaluation using Azure AI Foundry SDK
- Recognize the non-deterministic nature of LLM outputs

#### 📋 Lab Structure
1. **Introduction (15 min)**: The LLM evaluation problem
   - Why traditional software testing doesn't work for LLMs
   - Demo: Same prompt, different outputs
   - Business impact of poor LLM performance

2. **Core Concepts (15 min)**: Understanding evaluation metrics
   - Quality metrics: Relevance, Coherence, Fluency, Groundedness
   - Safety metrics: Harmful content, Bias detection
   - Performance metrics: Latency, Token usage, Cost analysis

3. **Hands-On: Basic Evaluation (25 min)**
   - Set up Azure AI Foundry evaluation environment
   - Create your first evaluation dataset (Q&A pairs)
   - Run built-in evaluators programmatically
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

### Lab 2: Scaling LLM Evaluations
**Notebook**: `lab2_scaling_evaluations.ipynb`
**Duration**: 60 minutes

#### 🎯 Learning Objectives
- Create and manage large-scale evaluation datasets
- Implement batch evaluation workflows
- Compare multiple models and prompts systematically
- Generate synthetic evaluation data

#### 📋 Lab Structure
1. **Dataset Strategies (15 min)**: Building quality evaluation sets
   - Synthetic data generation using LLMs
   - Curating real-world datasets
   - Data quality best practices
   - Evaluation dataset formats and schemas

2. **Hands-On: Dataset Generation (20 min)**
   - Use Azure OpenAI to generate synthetic Q&A pairs
   - Create domain-specific test scenarios
   - Implement data validation and quality checks

3. **Hands-On: Batch Evaluation (20 min)**
   - Set up multi-metric evaluation pipelines
   - Compare GPT-3.5 vs GPT-4 performance
   - Analyze cost vs. quality trade-offs
   - Generate evaluation reports

4. **Best Practices (5 min)**: Scaling patterns and optimization

#### 🔧 Technical Components
```python
# Key patterns you'll master
from azure.ai.evaluation.simulator import Simulator
from azure.ai.evaluation import evaluate

# Synthetic data generation
simulator = Simulator(model_config=config)
synthetic_data = await simulator(target=app, conversation_turns=turns)

# Multi-model comparison
results = evaluate(
    data=dataset,
    evaluators=multi_evaluator_config,
    model_configs={"gpt35": config1, "gpt4": config2}
)
```

#### 📈 Success Metrics
- Generate 50+ synthetic evaluation cases
- Successfully compare multiple models
- Understand scaling challenges and solutions

---

### Lab 3: Custom Evaluators & Domain-Specific Evaluation
**Notebook**: `lab3_custom_evaluators.ipynb`
**Duration**: 60 minutes

#### 🎯 Learning Objectives
- Build custom evaluation logic for specific use cases
- Implement domain-specific metrics (healthcare, legal, financial)
- Combine rule-based and AI-assisted evaluation approaches
- Handle complex, multi-dimensional evaluation scenarios

#### 📋 Lab Structure
1. **Custom Evaluator Patterns (15 min)**: Design approaches
   - Rule-based evaluators: Pattern matching, validation rules
   - LLM-based evaluators: Using models to judge outputs
   - Hybrid approaches: Combining multiple evaluation methods
   - Domain-specific requirements and compliance considerations

2. **Hands-On: Rule-Based Evaluators (15 min)**
   - Build medical terminology compliance checker
   - Implement safety requirement validation
   - Create structured output format validators

3. **Hands-On: LLM-Based Evaluators (25 min)**
   - Design expert-level evaluation prompts
   - Implement clinical accuracy assessment
   - Create rationale quality evaluators
   - Handle evaluation consistency and reliability

4. **Integration Patterns (5 min)**: Combining custom and built-in evaluators

#### 🔧 Technical Components
```python
# Advanced patterns you'll implement
class MedicalAccuracyEvaluator(EvaluatorBase):
    def __init__(self, policy_db, compliance_rules):
        self.policies = policy_db
        self.rules = compliance_rules
    
    def __call__(self, *, query: str, response: str, context: str, **kwargs):
        # Custom evaluation logic
        return evaluation_results

# LLM-assisted evaluation
class ExpertJudgeEvaluator(EvaluatorBase):
    def __call__(self, **kwargs):
        expert_prompt = self.build_evaluation_prompt(**kwargs)
        judgment = self.llm_client.complete(expert_prompt)
        return self.parse_judgment(judgment)
```

#### 📈 Success Metrics
- Build 2+ custom evaluators
- Understand when to use different evaluation approaches
- Successfully integrate custom logic with existing frameworks

---

### Lab 4: Enterprise Evaluation Frameworks
**Notebook**: `lab4_enterprise_frameworks.ipynb`
**Duration**: 60 minutes

#### 🎯 Learning Objectives
- Understand configuration-driven evaluation architecture
- Implement AutoAuth-style evaluation patterns
- Integrate with Azure AI Foundry for production monitoring
- Design scalable evaluation systems for enterprise use

#### 📋 Lab Structure
1. **Architecture Deep Dive (15 min)**: Enterprise evaluation systems
   - AutoAuth evaluation framework analysis
   - Configuration vs. implementation separation
   - Monitoring and observability patterns
   - CI/CD integration approaches

2. **Hands-On: Configuration-Driven Framework (25 min)**
   - Implement YAML-based evaluation configs
   - Build modular evaluator system
   - Create evaluation pipeline orchestration
   - Handle dynamic evaluator loading

3. **Hands-On: Production Integration (15 min)**
   - Connect to Azure AI Foundry for logging
   - Set up automated evaluation triggers
   - Implement evaluation result dashboards
   - Configure alerting and monitoring

4. **Deployment Patterns (5 min)**: Taking evaluations to production

#### 🔧 Technical Components
```yaml
# Configuration-driven evaluation (evaluation_config.yaml)
evaluation_suite:
  name: "healthcare_pa_evaluation"
  version: "1.0"
  
evaluation_cases:
  - name: "clinical_accuracy"
    evaluator: "ClinicalAccuracyEvaluator"
    parameters:
      medical_db: "ICD10_codes"
      confidence_threshold: 0.85
    metrics:
      - accuracy
      - precision
      - clinical_safety_score
      
  - name: "policy_compliance"
    evaluator: "PolicyComplianceEvaluator"
    parameters:
      policy_version: "2024_Q1"
      compliance_rules: "medicare_guidelines"
```

```python
# Enterprise framework implementation
class ConfigurableEvaluationPipeline:
    def __init__(self, config_path: str, ai_foundry_client):
        self.config = self.load_config(config_path)
        self.foundry_client = ai_foundry_client
        self.evaluators = self.initialize_evaluators()
    
    def run_evaluation_suite(self, dataset_path: str) -> Dict:
        results = {}
        for case in self.config['evaluation_cases']:
            case_results = self.run_evaluation_case(case, dataset_path)
            results[case['name']] = case_results
            self.log_to_foundry(case['name'], case_results)
        return results
```

#### 📈 Success Metrics
- Implement configuration-driven evaluation system
- Successfully integrate with Azure AI Foundry
- Understand enterprise deployment considerations
- Can adapt patterns to other domains

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
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Configure Azure Resources
```bash
# Set up environment variables
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_AI_FOUNDRY_PROJECT_NAME="your-project"

# Or create .env file with these values
```

### 2. Verify Setup
Run the setup verification notebook:
```bash
jupyter notebook setup_verification.ipynb
```

### 3. Start with Lab 1
```bash
jupyter notebook lab1_evaluation_fundamentals.ipynb
```

---

## 📁 Repository Structure

```
llm-evaluations-workshop/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup_verification.ipynb           # Environment setup verification
├── .env.template                      # Environment variables template
│
├── lab1_evaluation_fundamentals/
│   ├── lab1_evaluation_fundamentals.ipynb
│   ├── data/
│   │   ├── sample_qa_pairs.jsonl
│   │   └── evaluation_results.json
│   └── utils/
│       └── lab1_helpers.py
│
├── lab2_scaling_evaluations/
│   ├── lab2_scaling_evaluations.ipynb
│   ├── data/
│   │   ├── synthetic_dataset.jsonl
│   │   └── model_comparison_results.json
│   └── utils/
│       ├── data_generator.py
│       └── evaluation_pipeline.py
│
├── lab3_custom_evaluators/
│   ├── lab3_custom_evaluators.ipynb
│   ├── evaluators/
│   │   ├── medical_evaluators.py
│   │   ├── rule_based_evaluators.py
│   │   └── llm_assisted_evaluators.py
│   └── data/
│       └── domain_specific_tests.jsonl
│
├── lab4_enterprise_frameworks/
│   ├── lab4_enterprise_frameworks.ipynb
│   ├── config/
│   │   ├── evaluation_config.yaml
│   │   └── production_config.yaml
│   ├── framework/
│   │   ├── configurable_pipeline.py
│   │   ├── evaluator_registry.py
│   │   └── foundry_integration.py
│   └── examples/
│       ├── healthcare_evaluation.yaml
│       └── autoauth_adaptation.py
│
├── shared_utils/
│   ├── azure_clients.py               # Azure service clients
│   ├── evaluation_helpers.py          # Common evaluation utilities
│   └── data_utils.py                  # Dataset manipulation utilities
│
└── docs/
    ├── evaluation_metrics_guide.md    # Detailed metrics documentation
    ├── azure_setup_guide.md          # Azure resource setup
    ├── troubleshooting.md             # Common issues and solutions
    └── advanced_topics.md             # Beyond the workshop content
```

---

## 🎯 Workshop Outcomes

By the end of this workshop series, you will be able to:

### Technical Skills
- ✅ Implement comprehensive LLM evaluation pipelines
- ✅ Create custom evaluators for domain-specific requirements  
- ✅ Design configuration-driven evaluation frameworks
- ✅ Integrate evaluations with Azure AI Foundry for production monitoring
- ✅ Apply enterprise-grade evaluation patterns to your projects

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
- **Evaluation Patterns**: Check out the `examples/` directory for additional use cases

### Next Steps
After completing this workshop, consider exploring:
- **Advanced RAG Evaluation**: Specialized patterns for retrieval-augmented generation
- **Multi-modal Evaluation**: Evaluating vision and audio capabilities
- **A/B Testing for LLMs**: Statistical approaches to model comparison
- **Production Monitoring**: Real-time evaluation and alerting systems

---

## 🤝 Contributing

We welcome contributions to improve this workshop! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Reporting issues or bugs
- Suggesting new lab exercises
- Adding evaluation patterns
- Improving documentation

---

## 📄 License

This workshop is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🆘 Need Help?

- **Technical Issues**: Open an issue in this repository
- **Azure Setup Problems**: Check our [Azure Setup Guide](docs/azure_setup_guide.md)
- **Evaluation Questions**: Review the [Troubleshooting Guide](docs/troubleshooting.md)
- **Advanced Topics**: Explore [Advanced Topics Documentation](docs/advanced_topics.md)