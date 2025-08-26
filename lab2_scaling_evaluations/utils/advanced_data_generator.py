"""
Advanced data generation utilities for Lab 2 with AI Foundry focus.
"""

import json
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ComplexityLevel(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class QuestionType(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    CREATIVE = "creative"

@dataclass
class DatasetGenerationConfig:
    """Configuration for dataset generation."""
    domain: str
    num_pairs: int
    complexity_distribution: Dict[ComplexityLevel, float]
    question_type_distribution: Dict[QuestionType, float]
    include_context: bool = True
    include_metadata: bool = True
    foundry_ready: bool = True

class EnhancedDatasetGenerator:
    """Advanced dataset generator optimized for AI Foundry integration."""
    
    # Production-ready domain templates
    DOMAIN_TEMPLATES = {
        "customer_support": {
            "description": "Customer service and support scenarios",
            "context_types": ["product_info", "policy_info", "troubleshooting"],
            "sample_queries": [
                "How do I return a product?",
                "What is your refund policy?",
                "My account is locked, how do I unlock it?"
            ]
        },
        "technical_documentation": {
            "description": "Technical documentation and API guidance",
            "context_types": ["api_docs", "implementation_guides", "troubleshooting"],
            "sample_queries": [
                "How do I authenticate API requests?",
                "What are the rate limits for this endpoint?",
                "How do I handle pagination in the response?"
            ]
        },
        "healthcare_information": {
            "description": "General healthcare information and guidance",
            "context_types": ["medical_info", "wellness_guidance", "procedures"],
            "sample_queries": [
                "What are the symptoms of diabetes?",
                "How often should I get a health checkup?",
                "What is the recommended daily exercise?"
            ]
        },
        "business_analysis": {
            "description": "Business analysis and decision support",
            "context_types": ["market_data", "financial_info", "strategic_planning"],
            "sample_queries": [
                "What factors should I consider for market entry?",
                "How do I calculate ROI for this investment?",
                "What are the key performance indicators for retail?"
            ]
        },
        "legal_information": {
            "description": "General legal information and guidance",
            "context_types": ["regulations", "compliance", "procedures"],
            "sample_queries": [
                "What are the requirements for GDPR compliance?",
                "How do I file a trademark application?",
                "What are the key elements of a contract?"
            ]
        }
    }
    
    def __init__(self, azure_client, deployment_name: str, foundry_ready: bool = True):
        self.client = azure_client
        self.deployment = deployment_name
        self.foundry_ready = foundry_ready
        self.generation_history = []
        self.error_count = 0
        self.max_retries = 3
    
    def generate_production_dataset(self, 
                                  config: DatasetGenerationConfig) -> List[Dict[str, Any]]:
        """Generate a production-ready dataset with specified configuration."""
        
        print(f"🏭 Generating production dataset for domain: {config.domain}")
        print(f"📊 Target size: {config.num_pairs} pairs")
        print(f"🎯 Complexity levels: {list(config.complexity_distribution.keys())}")
        print(f"❓ Question types: {list(config.question_type_distribution.keys())}")
        
        dataset = []
        generation_start = time.time()
        
        # Calculate pairs per complexity level
        complexity_pairs = self._calculate_distribution(
            config.num_pairs, config.complexity_distribution
        )
        
        for complexity, pair_count in complexity_pairs.items():
            if pair_count == 0:
                continue
                
            print(f"   📝 Generating {pair_count} {complexity.value} level pairs...")
            
            complexity_data = self._generate_complexity_batch(
                config.domain,
                complexity,
                pair_count,
                config.question_type_distribution,
                config.include_context
            )
            
            dataset.extend(complexity_data)
            print(f"   ✅ Generated {len(complexity_data)} pairs for {complexity.value}")
        
        # Add metadata if requested
        if config.include_metadata:
            dataset = self._add_production_metadata(dataset, config)
        
        # Record generation statistics
        generation_time = time.time() - generation_start
        self._record_generation_stats(config, dataset, generation_time)
        
        print(f"✅ Production dataset complete: {len(dataset)} total pairs")
        print(f"⏱️ Generation time: {generation_time:.1f} seconds")
        
        return dataset
    
    def _calculate_distribution(self, total: int, distribution: Dict) -> Dict:
        """Calculate actual counts based on distribution percentages."""
        
        result = {}
        remaining = total
        
        # Sort by value to handle rounding properly
        sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        
        for item, percentage in sorted_items[:-1]:
            count = int(total * percentage)
            result[item] = count
            remaining -= count
        
        # Give remaining to the last item
        if sorted_items:
            result[sorted_items[-1][0]] = remaining
        
        return result
    
    def _generate_complexity_batch(self, 
                                 domain: str,
                                 complexity: ComplexityLevel, 
                                 pair_count: int,
                                 question_types: Dict[QuestionType, float],
                                 include_context: bool) -> List[Dict[str, Any]]:
        """Generate a batch of Q&A pairs for a specific complexity level."""
        
        domain_info = self.DOMAIN_TEMPLATES.get(domain, {
            "description": f"{domain} domain",
            "context_types": ["general"],
            "sample_queries": [f"Sample query for {domain}"]
        })
        
        generation_prompt = self._build_generation_prompt(
            domain, domain_info, complexity, pair_count, question_types, include_context
        )
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert evaluation dataset generator. Always respond with valid JSON arrays containing properly formatted Q&A pairs."
                        },
                        {"role": "user", "content": generation_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=3000
                )
                
                generated_text = response.choices[0].message.content
                pairs = self._extract_and_validate_pairs(generated_text, domain, complexity)
                
                if pairs:
                    return pairs[:pair_count]  # Ensure we don't exceed requested count
                else:
                    print(f"   ⚠️ Attempt {attempt + 1}: No valid pairs extracted")
                    
            except Exception as e:
                print(f"   ❌ Attempt {attempt + 1} failed: {e}")
                self.error_count += 1
        
        # Fallback: create sample pairs
        return self._create_fallback_pairs(domain, complexity, pair_count)
    
    def _build_generation_prompt(self, 
                               domain: str,
                               domain_info: Dict,
                               complexity: ComplexityLevel,
                               pair_count: int,
                               question_types: Dict,
                               include_context: bool) -> str:
        """Build a comprehensive prompt for data generation."""
        
        question_type_list = list(question_types.keys())
        
        prompt = f"""
Generate {pair_count} high-quality question-answer pairs for evaluation purposes.

DOMAIN: {domain}
DESCRIPTION: {domain_info.get('description', 'General domain')}
COMPLEXITY: {complexity.value}
QUESTION TYPES: Include a mix of {[qt.value for qt in question_type_list]}

REQUIREMENTS:
1. Questions should be realistic and practical for the {domain} domain
2. Answers should be accurate, comprehensive, and appropriate for {complexity.value} complexity
3. Each pair must be unique and add value to the evaluation dataset
4. Questions should vary in structure and approach
5. Answers should demonstrate appropriate depth for the complexity level

FORMAT: Return a JSON array where each object has this exact structure:
{{
    "query": "Your question here",
    "response": "Detailed answer here",
    "context": "Background information or relevant context",
    "domain": "{domain}",
    "complexity": "{complexity.value}",
    "question_type": "one of: {[qt.value for qt in question_type_list]}",
    "metadata": {{
        "estimated_tokens": 100,
        "difficulty_score": 1-10,
        "requires_expertise": true/false
    }}
}}

EXAMPLES for {complexity.value} level:
"""
        
        # Add complexity-specific examples
        if complexity == ComplexityLevel.BASIC:
            prompt += """
- Simple, direct questions requiring factual answers
- Clear, straightforward responses
- Common scenarios users might encounter
"""
        elif complexity == ComplexityLevel.INTERMEDIATE:
            prompt += """
- Questions requiring some analysis or multi-step thinking
- Answers that explain processes or relationships
- Moderate domain knowledge required
"""
        elif complexity == ComplexityLevel.ADVANCED:
            prompt += """
- Complex questions requiring deep analysis
- Comprehensive answers with multiple considerations
- High domain expertise required
"""
        
        prompt += f"\nGenerate exactly {pair_count} pairs in JSON array format:"
        
        return prompt
    
    def _extract_and_validate_pairs(self, 
                                  text: str, 
                                  domain: str, 
                                  complexity: ComplexityLevel) -> List[Dict[str, Any]]:
        """Extract and validate JSON pairs from generated text."""
        
        pairs = []
        
        # Try to find JSON array first
        json_array_pattern = r'\[[\s\S]*\]'
        array_matches = re.findall(json_array_pattern, text)
        
        for match in array_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    for item in parsed:
                        if self._validate_pair(item, domain, complexity):
                            pairs.append(item)
            except json.JSONDecodeError:
                continue
        
        # If no array found, try individual objects
        if not pairs:
            json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            object_matches = re.findall(json_object_pattern, text, re.DOTALL)
            
            for match in object_matches:
                try:
                    parsed = json.loads(match)
                    if self._validate_pair(parsed, domain, complexity):
                        pairs.append(parsed)
                except json.JSONDecodeError:
                    continue
        
        return pairs
    
    def _validate_pair(self, pair: Dict[str, Any], domain: str, complexity: ComplexityLevel) -> bool:
        """Validate a generated Q&A pair."""
        
        required_fields = ['query', 'response']
        
        # Check required fields
        for field in required_fields:
            if field not in pair or not isinstance(pair[field], str) or not pair[field].strip():
                return False
        
        # Check length requirements
        if len(pair['query']) < 10 or len(pair['response']) < 20:
            return False
        
        # Ensure domain and complexity are set
        pair['domain'] = domain
        pair['complexity'] = complexity.value
        
        # Add context if missing
        if 'context' not in pair or not pair['context']:
            pair['context'] = f"General context for {domain} domain question"
        
        # Add question type if missing
        if 'question_type' not in pair:
            pair['question_type'] = QuestionType.FACTUAL.value
        
        return True
    
    def _create_fallback_pairs(self, 
                             domain: str, 
                             complexity: ComplexityLevel, 
                             pair_count: int) -> List[Dict[str, Any]]:
        """Create fallback pairs when generation fails."""
        
        print(f"   ⚠️ Using fallback generation for {pair_count} pairs")
        
        fallback_pairs = []
        domain_info = self.DOMAIN_TEMPLATES.get(domain, {"sample_queries": ["Sample query"]})
        
        for i in range(pair_count):
            pair = {
                "query": f"Sample {complexity.value} question {i+1} for {domain}",
                "response": f"This is a sample {complexity.value} level answer for a {domain} question. "
                           f"It demonstrates the expected depth and detail for {complexity.value} complexity.",
                "context": f"Sample context for {domain} domain evaluation",
                "domain": domain,
                "complexity": complexity.value,
                "question_type": QuestionType.FACTUAL.value,
                "metadata": {
                    "generated_via": "fallback",
                    "estimated_tokens": 50,
                    "difficulty_score": 3 if complexity == ComplexityLevel.BASIC else 
                                     6 if complexity == ComplexityLevel.INTERMEDIATE else 8
                }
            }
            fallback_pairs.append(pair)
        
        return fallback_pairs
    
    def _add_production_metadata(self, 
                               dataset: List[Dict[str, Any]], 
                               config: DatasetGenerationConfig) -> List[Dict[str, Any]]:
        """Add production-ready metadata to dataset."""
        
        enhanced_dataset = []
        
        for i, item in enumerate(dataset):
            enhanced_item = {
                **item,
                "id": f"{config.domain}_{i+1:04d}",
                "generation_timestamp": datetime.now().isoformat(),
                "foundry_ready": self.foundry_ready,
                "evaluation_metadata": {
                    "priority": "high" if item.get('complexity') == 'advanced' else "medium",
                    "estimated_evaluation_time_ms": 2000 if item.get('complexity') == 'advanced' else 1000,
                    "requires_human_review": item.get('complexity') == 'expert'
                }
            }
            
            if self.foundry_ready:
                enhanced_item["ai_foundry_tags"] = [
                    config.domain,
                    item.get('complexity', 'basic'),
                    item.get('question_type', 'factual'),
                    f"lab2_generated_{datetime.now().strftime('%Y%m%d')}"
                ]
            
            enhanced_dataset.append(enhanced_item)
        
        return enhanced_dataset
    
    def _record_generation_stats(self, 
                               config: DatasetGenerationConfig,
                               dataset: List[Dict[str, Any]], 
                               generation_time: float):
        """Record statistics about the generation process."""
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'domain': config.domain,
            'requested_pairs': config.num_pairs,
            'generated_pairs': len(dataset),
            'generation_time_seconds': generation_time,
            'complexity_distribution': {
                level.value: len([item for item in dataset if item.get('complexity') == level.value])
                for level in ComplexityLevel
            },
            'question_type_distribution': {
                qtype.value: len([item for item in dataset if item.get('question_type') == qtype.value])
                for qtype in QuestionType
            },
            'error_count': self.error_count,
            'foundry_ready': self.foundry_ready
        }
        
        self.generation_history.append(stats)
    
    def create_multi_domain_production_dataset(self, 
                                             domain_configs: List[DatasetGenerationConfig]) -> List[Dict[str, Any]]:
        """Create a comprehensive production dataset spanning multiple domains."""
        
        print(f"🌍 Creating multi-domain production dataset")
        print(f"📊 Domains: {[config.domain for config in domain_configs]}")
        print(f"🔢 Total target pairs: {sum(config.num_pairs for config in domain_configs)}")
        
        all_data = []
        start_time = time.time()
        
        for config in domain_configs:
            print(f"\n🔄 Processing domain: {config.domain}")
            domain_data = self.generate_production_dataset(config)
            all_data.extend(domain_data)
            print(f"✅ Completed {config.domain}: {len(domain_data)} pairs")
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 Multi-domain dataset complete!")
        print(f"📊 Total pairs: {len(all_data)}")
        print(f"⏱️ Total time: {total_time:.1f} seconds")
        print(f"🏢 AI Foundry ready: {'Yes' if self.foundry_ready else 'No'}")
        
        return all_data
    
    def get_generation_report(self) -> str:
        """Generate a comprehensive report of all generation activities."""
        
        if not self.generation_history:
            return "No generation activities recorded."
        
        total_requested = sum(stats['requested_pairs'] for stats in self.generation_history)
        total_generated = sum(stats['generated_pairs'] for stats in self.generation_history)
        total_time = sum(stats['generation_time_seconds'] for stats in self.generation_history)
        
        report = []
        report.append("📊 DATA GENERATION REPORT")
        report.append("=" * 30)
        report.append(f"Total sessions: {len(self.generation_history)}")
        report.append(f"Total pairs requested: {total_requested}")
        report.append(f"Total pairs generated: {total_generated}")
        report.append(f"Success rate: {(total_generated/total_requested)*100:.1f}%")
        report.append(f"Total generation time: {total_time:.1f} seconds")
        report.append(f"Average time per pair: {(total_time/total_generated):.2f} seconds" if total_generated > 0 else "N/A")
        report.append(f"Total errors: {self.error_count}")
        
        # Domain breakdown
        domains = {}
        for stats in self.generation_history:
            domains[stats['domain']] = domains.get(stats['domain'], 0) + stats['generated_pairs']
        
        report.append(f"\n🌍 DOMAIN BREAKDOWN:")
        for domain, count in domains.items():
            report.append(f"   {domain}: {count} pairs")
        
        return "\n".join(report)


# Factory functions for common configurations
def create_customer_support_config(num_pairs: int = 20) -> DatasetGenerationConfig:
    """Create a configuration optimized for customer support scenarios."""
    
    return DatasetGenerationConfig(
        domain="customer_support",
        num_pairs=num_pairs,
        complexity_distribution={
            ComplexityLevel.BASIC: 0.4,
            ComplexityLevel.INTERMEDIATE: 0.4,
            ComplexityLevel.ADVANCED: 0.2
        },
        question_type_distribution={
            QuestionType.PROCEDURAL: 0.5,
            QuestionType.FACTUAL: 0.3,
            QuestionType.ANALYTICAL: 0.2
        },
        foundry_ready=True
    )

def create_technical_docs_config(num_pairs: int = 20) -> DatasetGenerationConfig:
    """Create a configuration optimized for technical documentation."""
    
    return DatasetGenerationConfig(
        domain="technical_documentation",
        num_pairs=num_pairs,
        complexity_distribution={
            ComplexityLevel.INTERMEDIATE: 0.4,
            ComplexityLevel.ADVANCED: 0.4,
            ComplexityLevel.BASIC: 0.2
        },
        question_type_distribution={
            QuestionType.PROCEDURAL: 0.4,
            QuestionType.ANALYTICAL: 0.3,
            QuestionType.FACTUAL: 0.3
        },
        foundry_ready=True
    )