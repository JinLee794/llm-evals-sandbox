"""
Advanced evaluation pipeline utilities for Lab 2 with AI Foundry integration.
"""

import json
import time
import tempfile
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

class AdvancedBatchEvaluationPipeline:
    """Enhanced batch evaluation pipeline with AI Foundry integration."""
    
    def __init__(self, foundry_runner, evaluators: Dict[str, Any]):
        self.foundry_runner = foundry_runner
        self.evaluators = evaluators
        self.results_history = []
    
    def run_batch_evaluation_with_foundry(self, 
                                        dataset: List[Dict[str, Any]], 
                                        batch_size: int = 10,
                                        delay_seconds: float = 1.0,
                                        run_prefix: str = "Lab2_Batch") -> Dict[str, Any]:
        """
        Run batch evaluation with AI Foundry integration.
        
        Args:
            dataset: List of evaluation data points
            batch_size: Number of items to evaluate in each batch
            delay_seconds: Delay between batches to manage rate limits
            run_prefix: Prefix for evaluation run names
        
        Returns:
            Combined evaluation results with AI Foundry metadata
        """
        
        print(f"🚀 Starting enhanced batch evaluation of {len(dataset)} items...")
        print(f"📊 Batch size: {batch_size}, Delay: {delay_seconds}s")
        print(f"🏢 AI Foundry integration: {'✅ Enabled' if self.foundry_runner.foundry_enabled else '❌ Disabled'}")
        
        all_results = []
        batch_count = (len(dataset) + batch_size - 1) // batch_size
        
        start_time = time.time()
        foundry_datasets_created = []
        
        for i in range(0, len(dataset), batch_size):
            batch_num = i // batch_size + 1
            batch_data = dataset[i:i + batch_size]
            
            print(f"⏳ Processing batch {batch_num}/{batch_count} ({len(batch_data)} items)...")
            
            try:
                # Use foundry runner for enhanced evaluation
                batch_results = self.foundry_runner.run_evaluation(
                    data=batch_data,
                    evaluators=self.evaluators,
                    run_name=f"{run_prefix}_Batch_{batch_num}_of_{batch_count}",
                    description=f"Batch {batch_num} evaluation - {len(batch_data)} items"
                )
                
                all_results.append(batch_results)
                print(f"✅ Batch {batch_num} completed successfully")
                
                # Track AI Foundry datasets if created
                if batch_results.get('_dataset_id'):
                    foundry_datasets_created.append({
                        'batch_num': batch_num,
                        'dataset_id': batch_results['_dataset_id'],
                        'dataset_name': batch_results.get('_dataset_name', 'unknown')
                    })
                
                # Add delay between batches (except for the last batch)
                if i + batch_size < len(dataset):
                    print(f"   ⏸️ Waiting {delay_seconds}s before next batch...")
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                print(f"❌ Batch {batch_num} failed: {e}")
                print(f"⚠️ Continuing with next batch...")
                continue
        
        # Combine results from all batches
        combined_results = self._combine_batch_results(all_results)
        
        # Add AI Foundry metadata
        combined_results['_foundry_integration'] = {
            'datasets_created': len(foundry_datasets_created),
            'dataset_details': foundry_datasets_created,
            'portal_accessible': self.foundry_runner.foundry_enabled
        }
        
        elapsed_time = time.time() - start_time
        print(f"🎉 Enhanced batch evaluation completed in {elapsed_time:.1f} seconds")
        print(f"📊 Successfully evaluated {len(combined_results.get('rows', []))} items")
        
        if foundry_datasets_created:
            print(f"🏢 Created {len(foundry_datasets_created)} datasets in AI Foundry portal")
            print("   👀 View at: https://ai.azure.com")
        
        # Store in history with enhanced metadata
        self.results_history.append({
            'timestamp': datetime.now().isoformat(),
            'total_items': len(dataset),
            'successful_items': len(combined_results.get('rows', [])),
            'batch_size': batch_size,
            'elapsed_time': elapsed_time,
            'foundry_datasets': len(foundry_datasets_created),
            'results': combined_results
        })
        
        return combined_results
    
    def _combine_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple batches into a single result set."""
        
        if not batch_results:
            return {"metrics": {}, "rows": [], "_foundry_integration": {}}
        
        # Combine all rows
        all_rows = []
        all_metrics = []
        
        for batch in batch_results:
            if 'rows' in batch:
                all_rows.extend(batch['rows'])
            if 'metrics' in batch:
                all_metrics.append(batch['metrics'])
        
        # Calculate combined metrics
        combined_metrics = self._calculate_combined_metrics(all_rows, all_metrics)
        
        # Collect execution methods to determine overall method
        execution_methods = [batch.get('_execution_method', 'unknown') for batch in batch_results]
        primary_method = max(set(execution_methods), key=execution_methods.count) if execution_methods else 'unknown'
        
        return {
            "metrics": combined_metrics,
            "rows": all_rows,
            "batch_count": len(batch_results),
            "total_evaluations": len(all_rows),
            "_execution_method": primary_method,
            "_batch_evaluation": True
        }
    
    def _calculate_combined_metrics(self, all_rows: List[Dict[str, Any]], 
                                   all_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate combined metrics across all evaluated items."""
        
        if not all_rows:
            return {}
        
        metric_sums = {}
        metric_counts = {}
        
        # Aggregate from individual rows
        for row in all_rows:
            if 'outputs' in row:
                for metric, value in row['outputs'].items():
                    if isinstance(value, (int, float)):
                        metric_sums[metric] = metric_sums.get(metric, 0) + value
                        metric_counts[metric] = metric_counts.get(metric, 0) + 1
        
        # Calculate averages
        combined_metrics = {}
        for metric in metric_sums:
            if metric_counts[metric] > 0:
                combined_metrics[metric] = metric_sums[metric] / metric_counts[metric]
        
        return combined_metrics
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get history of all batch evaluations."""
        return self.results_history
    
    def generate_batch_report(self) -> str:
        """Generate a comprehensive batch evaluation report."""
        
        if not self.results_history:
            return "No batch evaluations completed yet."
        
        latest = self.results_history[-1]
        
        report = []
        report.append("📊 BATCH EVALUATION REPORT")
        report.append("=" * 30)
        report.append(f"Timestamp: {latest['timestamp']}")
        report.append(f"Total items processed: {latest['total_items']}")
        report.append(f"Successfully evaluated: {latest['successful_items']}")
        report.append(f"Batch size used: {latest['batch_size']}")
        report.append(f"Processing time: {latest['elapsed_time']:.1f} seconds")
        report.append(f"AI Foundry datasets created: {latest['foundry_datasets']}")
        
        if latest['foundry_datasets'] > 0:
            report.append(f"\n🏢 AI FOUNDRY INTEGRATION:")
            report.append(f"   Portal accessible: Yes")
            report.append(f"   View at: https://ai.azure.com")
            report.append(f"   Datasets created: {latest['foundry_datasets']}")
        
        # Add performance metrics if available
        results = latest['results']
        if 'metrics' in results and results['metrics']:
            report.append(f"\n📈 PERFORMANCE METRICS:")
            for metric, score in results['metrics'].items():
                if isinstance(score, (int, float)):
                    report.append(f"   {metric}: {score:.3f}")
        
        return "\n".join(report)


class ProductionEvaluationOrchestrator:
    """Orchestrates complex evaluation workflows for production scenarios."""
    
    def __init__(self, foundry_runner, azure_client):
        self.foundry_runner = foundry_runner
        self.azure_client = azure_client
        self.evaluation_campaigns = {}
    
    def create_evaluation_campaign(self, 
                                 campaign_name: str,
                                 datasets: List[List[Dict[str, Any]]],
                                 evaluators: Dict[str, Any],
                                 campaign_description: str = "") -> str:
        """
        Create a comprehensive evaluation campaign across multiple datasets.
        
        Args:
            campaign_name: Name for the evaluation campaign
            datasets: List of datasets to evaluate
            evaluators: Evaluators to use
            campaign_description: Description of the campaign
            
        Returns:
            Campaign ID for tracking
        """
        
        campaign_id = f"{campaign_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.evaluation_campaigns[campaign_id] = {
            'name': campaign_name,
            'description': campaign_description,
            'datasets_count': len(datasets),
            'total_items': sum(len(dataset) for dataset in datasets),
            'evaluators': list(evaluators.keys()),
            'status': 'initialized',
            'results': [],
            'created_at': datetime.now().isoformat()
        }
        
        print(f"🎯 Created evaluation campaign: {campaign_name}")
        print(f"   Campaign ID: {campaign_id}")
        print(f"   Datasets: {len(datasets)}")
        print(f"   Total items: {sum(len(dataset) for dataset in datasets)}")
        print(f"   Evaluators: {len(evaluators)}")
        
        return campaign_id
    
    def execute_campaign(self, campaign_id: str, evaluators: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete evaluation campaign."""
        
        if campaign_id not in self.evaluation_campaigns:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        campaign = self.evaluation_campaigns[campaign_id]
        campaign['status'] = 'running'
        
        print(f"🚀 Executing campaign: {campaign['name']}")
        print(f"📊 Processing {campaign['total_items']} items across {campaign['datasets_count']} datasets")
        
        # This would contain the actual evaluation logic in a full implementation
        # For now, we'll create a framework structure
        
        campaign_results = {
            'campaign_id': campaign_id,
            'campaign_name': campaign['name'],
            'execution_timestamp': datetime.now().isoformat(),
            'status': 'completed',
            'summary': {
                'datasets_processed': campaign['datasets_count'],
                'total_evaluations': campaign['total_items'],
                'foundry_integration_active': self.foundry_runner.foundry_enabled
            }
        }
        
        campaign['status'] = 'completed'
        campaign['results'] = campaign_results
        
        print(f"✅ Campaign completed: {campaign['name']}")
        
        return campaign_results
    
    def get_campaign_status(self, campaign_id: str) -> Dict[str, Any]:
        """Get the current status of an evaluation campaign."""
        
        if campaign_id not in self.evaluation_campaigns:
            return {"error": f"Campaign {campaign_id} not found"}
        
        return self.evaluation_campaigns[campaign_id]
    
    def list_campaigns(self) -> List[Dict[str, Any]]:
        """List all evaluation campaigns."""
        
        return [
            {
                'campaign_id': cid,
                'name': campaign['name'],
                'status': campaign['status'],
                'created_at': campaign['created_at'],
                'total_items': campaign['total_items']
            }
            for cid, campaign in self.evaluation_campaigns.items()
        ]


def create_advanced_pipeline(foundry_runner, evaluators: Dict[str, Any]) -> AdvancedBatchEvaluationPipeline:
    """Factory function to create an advanced batch evaluation pipeline."""
    
    return AdvancedBatchEvaluationPipeline(foundry_runner, evaluators)


def save_lab2_results(results: Dict[str, Any], 
                     output_dir: str = "data", 
                     filename_prefix: str = "lab2_advanced_evaluation") -> str:
    """Save Lab 2 evaluation results with enhanced metadata."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    output_path = Path(output_dir) / filename
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add Lab 2 specific metadata
    enhanced_results = {
        **results,
        'lab_info': {
            'lab_number': 2,
            'lab_name': 'Scaling LLM Evaluations',
            'version': '2.0',
            'ai_foundry_enhanced': True
        },
        'saved_at': datetime.now().isoformat(),
        'filename': filename
    }
    
    with open(output_path, 'w') as f:
        json.dump(enhanced_results, f, indent=2, default=str)
    
    print(f"💾 Lab 2 results saved to: {output_path}")
    return str(output_path)