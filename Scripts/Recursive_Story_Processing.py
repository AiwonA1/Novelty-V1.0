import logging
import sys
import os
import time
import asyncio
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Methods.story_manager import StoryManager
from Methods.Computational_Methods import (
    process_story_segment,
    ActiveInference,
    EnergyOptimizer,
    RecursiveProcessor
)
from Scripts.Performance_Monitor import SystemMonitor

class EnhancedStoryProcessor:
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.story_manager = StoryManager()
        self.active_inference = ActiveInference()
        self.energy_optimizer = EnergyOptimizer()
        self.recursive_processor = RecursiveProcessor()
        self.system_monitor = SystemMonitor()
        self.performance_data: List[Dict[str, float]] = []
        
        # Configure logging
        logging.basicConfig(
            filename='enhanced_story_processing.log',
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def process_story_with_monitoring(
        self, 
        story: str, 
        depth: int = 0
    ) -> Tuple[str, Dict[str, float]]:
        """Process story segment with performance monitoring."""
        
        start_time = time.time()
        metrics_before = self.system_monitor.collect_system_metrics()
        
        try:
            # Process the story segment
            processed_story = await self._process_segment(story, depth)
            
            # Collect performance metrics
            metrics_after = self.system_monitor.collect_system_metrics()
            processing_time = time.time() - start_time
            
            # Calculate metric deltas
            metrics_delta = {
                'processing_time': processing_time,
                'cpu_delta': metrics_after['cpu_usage'] - metrics_before['cpu_usage'],
                'memory_delta': metrics_after['memory_usage'] - metrics_before['memory_usage']
            }
            
            if 'gpu_memory' in metrics_after and 'gpu_memory' in metrics_before:
                metrics_delta['gpu_memory_delta'] = metrics_after['gpu_memory'] - metrics_before['gpu_memory']
            
            if 'power_draw' in metrics_after and 'power_draw' in metrics_before:
                metrics_delta['power_draw_delta'] = metrics_after['power_draw'] - metrics_before['power_draw']
            
            self.performance_data.append(metrics_delta)
            self.logger.info(f"Performance metrics at depth {depth}: {metrics_delta}")
            
            return processed_story, metrics_delta
            
        except Exception as e:
            self.logger.error(f"Error processing story at depth {depth}: {str(e)}")
            raise

    async def _process_segment(self, story: str, depth: int) -> str:
        """Internal method to process a story segment with active inference."""
        
        if depth > self.max_depth:
            self.logger.info(f"Maximum recursion depth {self.max_depth} reached.")
            return story

        try:
            # Apply active inference to predict optimal processing approach
            story_data = np.array([len(story)])  # Simple feature for demonstration
            self.active_inference.update_beliefs(story_data)
            
            # Process the story segment
            processed_story = process_story_segment(story, depth)
            
            # Track themes and character development
            self._analyze_and_track_story_elements(processed_story)
            
            # Optimize resource usage
            self.energy_optimizer.optimize_resources()
            
            # Recursively process with feedback
            feedback = self._generate_feedback(processed_story)
            refined_story = await self.recursive_processor.recursive_refine(
                processed_story, 
                feedback
            )
            
            return refined_story
            
        except Exception as e:
            self.logger.error(f"Error in _process_segment at depth {depth}: {str(e)}")
            raise

    def _analyze_and_track_story_elements(self, story: str):
        """Analyze and track story elements using StoryManager."""
        try:
            # Analyze sentiment
            sentiment = self.story_manager.analyze_sentiment(story)
            
            # Extract and track major themes
            # This is a simplified example - in practice, you'd want more sophisticated
            # theme extraction
            words = story.split()
            potential_themes = [word for word in words if len(word) > 5]
            for theme in potential_themes[:3]:  # Track top 3 longest words as themes
                self.story_manager.track_theme(theme)
            
            # Track plot progression
            self.story_manager.progress_plot(story[:100] + "...")  # First 100 chars
            
        except Exception as e:
            self.logger.error(f"Error analyzing story elements: {str(e)}")
            raise

    def _generate_feedback(self, story: str) -> str:
        """Generate feedback for story refinement."""
        try:
            # This is a simplified example - in practice, you'd want more sophisticated
            # feedback generation
            feedback = []
            
            # Length-based feedback
            if len(story) < 100:
                feedback.append("Consider expanding the story with more detail.")
            elif len(story) > 1000:
                feedback.append("Consider condensing some sections.")
            
            # Sentiment-based feedback
            sentiment = self.story_manager.analyze_sentiment(story)
            if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.8:
                feedback.append("Consider balancing the negative tone with positive elements.")
            
            return " ".join(feedback) if feedback else "Story structure looks good."
            
        except Exception as e:
            self.logger.error(f"Error generating feedback: {str(e)}")
            raise

    def visualize_performance(self):
        """Visualize collected performance metrics."""
        self.system_monitor.visualize_metrics('story_processing_performance.png')

async def main():
    try:
        processor = EnhancedStoryProcessor(max_depth=5)
        
        # Example story
        story = "Once upon a time in a digital realm, an AI began to tell a story..."
        
        processed_story, metrics = await processor.process_story_with_monitoring(story)
        
        # Save the processed story
        with open('processed_story.txt', 'w') as f:
            f.write(processed_story)
        
        # Generate performance visualization
        processor.visualize_performance()
        
        logging.info("Story processing completed successfully")
        logging.info(f"Final performance metrics: {metrics}")
        
    except Exception as e:
        logging.critical(f"Critical error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())