import sys
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Methods.story_manager import StoryManager

@dataclass
class ThemeMetrics:
    relevance: float
    consistency: float
    development: float
    impact: float

    def overall_score(self) -> float:
        return np.mean([
            self.relevance,
            self.consistency,
            self.development,
            self.impact
        ])

class StoryAnalysisEngine:
    def __init__(self):
        self.story_manager = StoryManager()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.theme_classifier = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased'
        )
        self.ner_pipeline = pipeline('ner', model='bert-base-uncased')
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        
        # Theme tracking
        self.theme_history: Dict[str, List[ThemeMetrics]] = defaultdict(list)
        self.character_arcs: Dict[str, List[Dict]] = defaultdict(list)
        self.plot_points: List[Dict] = []
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.theme_classifier.to(self.device)
        
        logging.basicConfig(
            filename='story_analysis.log',
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def analyze_story_segment(
        self, 
        segment: str,
        context: Optional[Dict] = None
    ) -> Dict[str, any]:
        """Perform comprehensive analysis of a story segment."""
        try:
            analysis = {
                'themes': await self._extract_themes(segment),
                'characters': await self._analyze_characters(segment),
                'plot_development': await self._analyze_plot(segment),
                'emotional_arc': await self._analyze_emotional_arc(segment),
                'narrative_coherence': await self._assess_coherence(segment, context),
                'style_metrics': await self._analyze_style(segment)
            }
            
            # Update tracking
            self._update_story_tracking(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in story analysis: {str(e)}")
            raise

    async def _extract_themes(self, text: str) -> Dict[str, ThemeMetrics]:
        """Extract and analyze themes from the text."""
        themes = {}
        
        try:
            # Tokenize and prepare text
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get theme predictions
            with torch.no_grad():
                outputs = self.theme_classifier(**inputs)
            
            # Process theme predictions
            theme_scores = torch.softmax(outputs.logits, dim=1)
            
            # Analyze each potential theme
            for theme_idx in torch.topk(theme_scores, k=5).indices[0]:
                theme_name = self.theme_classifier.config.id2label[theme_idx.item()]
                theme_score = theme_scores[0][theme_idx].item()
                
                # Calculate theme metrics
                metrics = ThemeMetrics(
                    relevance=theme_score,
                    consistency=self._calculate_theme_consistency(theme_name),
                    development=self._analyze_theme_development(theme_name),
                    impact=self._assess_theme_impact(theme_name, text)
                )
                
                themes[theme_name] = metrics
            
            return themes
            
        except Exception as e:
            self.logger.error(f"Error extracting themes: {str(e)}")
            raise

    async def _analyze_characters(self, text: str) -> Dict[str, Dict]:
        """Analyze character development and interactions."""
        try:
            characters = {}
            
            # Extract named entities
            entities = self.ner_pipeline(text)
            
            # Group by character names
            for entity in entities:
                if entity['entity'] == 'PER':  # Person entity
                    char_name = entity['word']
                    if char_name not in characters:
                        characters[char_name] = {
                            'mentions': [],
                            'sentiment_context': [],
                            'interactions': defaultdict(list)
                        }
                    
                    # Record mention context
                    context_start = max(0, entity['start'] - 50)
                    context_end = min(len(text), entity['end'] + 50)
                    context = text[context_start:context_end]
                    
                    # Analyze sentiment in context
                    sentiment = self.sentiment_analyzer(context)[0]
                    
                    characters[char_name]['mentions'].append({
                        'context': context,
                        'sentiment': sentiment
                    })
            
            # Analyze character interactions
            for char1 in characters:
                for char2 in characters:
                    if char1 != char2:
                        interactions = self._analyze_character_interaction(
                            text, char1, char2
                        )
                        if interactions:
                            characters[char1]['interactions'][char2].extend(interactions)
            
            return characters
            
        except Exception as e:
            self.logger.error(f"Error analyzing characters: {str(e)}")
            raise

    async def _analyze_plot(self, text: str) -> Dict[str, any]:
        """Analyze plot development and structure."""
        try:
            # Split into sentences for analysis
            sentences = text.split('.')
            plot_points = []
            
            current_state = {
                'tension': 0.0,
                'progress': 0.0,
                'complexity': 0.0
            }
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                    
                # Analyze sentence impact on plot
                sentiment = self.sentiment_analyzer(sentence)[0]
                
                # Update plot metrics
                current_state['tension'] = self._calculate_tension(
                    sentence, current_state['tension']
                )
                current_state['progress'] = i / len(sentences)
                current_state['complexity'] = self._calculate_complexity(
                    sentence, current_state['complexity']
                )
                
                plot_points.append({
                    'text': sentence.strip(),
                    'position': i / len(sentences),
                    'sentiment': sentiment,
                    'state': current_state.copy()
                })
            
            return {
                'plot_points': plot_points,
                'structure_analysis': self._analyze_plot_structure(plot_points),
                'pacing': self._analyze_pacing(plot_points)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing plot: {str(e)}")
            raise

    def _calculate_tension(self, text: str, current_tension: float) -> float:
        """Calculate narrative tension based on text content."""
        # This is a simplified example - in practice, you'd want more sophisticated
        # tension analysis
        sentiment = self.sentiment_analyzer(text)[0]
        
        # Adjust tension based on sentiment
        if sentiment['label'] == 'NEGATIVE':
            tension_delta = 0.1 * sentiment['score']
        else:
            tension_delta = -0.05 * sentiment['score']
            
        return max(0.0, min(1.0, current_tension + tension_delta))

    def _calculate_complexity(self, text: str, current_complexity: float) -> float:
        """Calculate narrative complexity based on text content."""
        # This is a simplified example - in practice, you'd want more sophisticated
        # complexity analysis
        words = text.split()
        unique_words = len(set(words))
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        complexity_score = (unique_words / len(words) if words else 0) * (avg_word_length / 10)
        
        return max(0.0, min(1.0, (current_complexity + complexity_score) / 2))

    def _analyze_plot_structure(self, plot_points: List[Dict]) -> Dict[str, float]:
        """Analyze the overall plot structure."""
        return {
            'coherence': self._calculate_coherence(plot_points),
            'pacing': self._calculate_pacing(plot_points),
            'complexity': np.mean([p['state']['complexity'] for p in plot_points])
        }

    def _calculate_coherence(self, plot_points: List[Dict]) -> float:
        """Calculate narrative coherence score."""
        if not plot_points:
            return 0.0
            
        # Analyze sentiment flow
        sentiments = [p['sentiment']['score'] for p in plot_points]
        
        # Calculate smoothness of sentiment transitions
        sentiment_diffs = np.diff(sentiments)
        coherence_score = 1.0 - (np.std(sentiment_diffs) / 2.0)
        
        return max(0.0, min(1.0, coherence_score))

    def _calculate_pacing(self, plot_points: List[Dict]) -> float:
        """Calculate story pacing score."""
        if not plot_points:
            return 0.0
            
        # Analyze tension progression
        tensions = [p['state']['tension'] for p in plot_points]
        
        # Calculate pacing based on tension changes
        tension_changes = np.diff(tensions)
        pacing_score = np.mean(np.abs(tension_changes)) * 5.0  # Scale to 0-1
        
        return max(0.0, min(1.0, pacing_score))

    def _update_story_tracking(self, analysis: Dict[str, any]):
        """Update story tracking with new analysis results."""
        # Update theme history
        for theme, metrics in analysis['themes'].items():
            self.theme_history[theme].append(metrics)
        
        # Update character arcs
        for char, data in analysis['characters'].items():
            self.character_arcs[char].append(data)
        
        # Update plot points
        self.plot_points.extend(analysis['plot_development']['plot_points'])

async def main():
    try:
        analyzer = StoryAnalysisEngine()
        
        # Example story segment
        story = """In the depths of the quantum realm, where reality itself seemed 
        to flicker between states of existence, Dr. Sarah Chen made a discovery 
        that would challenge everything she thought she knew about consciousness..."""
        
        # Perform analysis
        analysis = await analyzer.analyze_story_segment(story)
        
        # Log results
        logging.info("Story analysis completed successfully")
        logging.info(f"Analysis results: {analysis}")
        
    except Exception as e:
        logging.critical(f"Critical error in story analysis: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 