import sys
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
import torch
import spacy
from transformers import pipeline
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy.signal import savgol_filter
from dataclasses import dataclass
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Scripts.Story_Analysis_Engine import StoryAnalysisEngine, ThemeMetrics

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    plot_width: int = 1200
    plot_height: int = 800
    theme_colors: Dict[str, str] = None
    character_colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.theme_colors is None:
            self.theme_colors = {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'tertiary': '#2ca02c',
                'quaternary': '#d62728',
                'quinary': '#9467bd'
            }
        if self.character_colors is None:
            self.character_colors = px.colors.qualitative.Set3

class StoryVisualizer:
    def __init__(self):
        self.story_analyzer = StoryAnalysisEngine()
        self.config = VisualizationConfig()
        self.nlp = spacy.load('en_core_web_sm')
        self.vader = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(max_features=100)
        
        logging.basicConfig(
            filename='story_visualization.log',
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def generate_comprehensive_visualization(
        self, 
        story: str,
        output_dir: str = 'visualizations'
    ) -> Dict[str, str]:
        """Generate comprehensive story visualizations."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Analyze story
            analysis = await self.story_analyzer.analyze_story_segment(story)
            
            # Generate visualizations
            visualizations = {}
            
            # 1. Emotional Arc
            emotional_arc = self._create_emotional_arc(story)
            emotional_arc.write_html(f"{output_dir}/emotional_arc.html")
            visualizations['emotional_arc'] = f"{output_dir}/emotional_arc.html"
            
            # 2. Character Network
            char_network = self._create_character_network(analysis['characters'])
            char_network.write_html(f"{output_dir}/character_network.html")
            visualizations['character_network'] = f"{output_dir}/character_network.html"
            
            # 3. Theme Evolution
            theme_evolution = self._create_theme_evolution(analysis['themes'])
            theme_evolution.write_html(f"{output_dir}/theme_evolution.html")
            visualizations['theme_evolution'] = f"{output_dir}/theme_evolution.html"
            
            # 4. Narrative Structure
            narrative_structure = self._create_narrative_structure(analysis)
            narrative_structure.write_html(f"{output_dir}/narrative_structure.html")
            visualizations['narrative_structure'] = f"{output_dir}/narrative_structure.html"
            
            # 5. Style Analysis
            style_analysis = self._create_style_analysis(story)
            style_analysis.write_html(f"{output_dir}/style_analysis.html")
            visualizations['style_analysis'] = f"{output_dir}/style_analysis.html"
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            raise

    def _create_emotional_arc(self, story: str) -> go.Figure:
        """Create an interactive emotional arc visualization."""
        try:
            # Split into sentences
            sentences = sent_tokenize(story)
            
            # Calculate emotional values
            emotions = []
            for sentence in sentences:
                scores = self.vader.polarity_scores(sentence)
                emotions.append(scores['compound'])
            
            # Smooth the emotional arc
            x = np.linspace(0, 1, len(emotions))
            emotions_smooth = savgol_filter(emotions, 
                                         min(7, len(emotions)), 3)
            
            # Create interactive plot
            fig = go.Figure()
            
            # Add raw emotional values
            fig.add_trace(go.Scatter(
                x=x,
                y=emotions,
                mode='markers',
                name='Raw Emotional Values',
                marker=dict(size=8),
                hovertext=sentences
            ))
            
            # Add smoothed emotional arc
            fig.add_trace(go.Scatter(
                x=x,
                y=emotions_smooth,
                mode='lines',
                name='Emotional Arc',
                line=dict(width=3, color=self.config.theme_colors['primary'])
            ))
            
            # Update layout
            fig.update_layout(
                title='Story Emotional Arc',
                xaxis_title='Story Progress',
                yaxis_title='Emotional Valence',
                width=self.config.plot_width,
                height=self.config.plot_height,
                hovermode='closest',
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating emotional arc: {str(e)}")
            raise

    def _create_character_network(self, character_data: Dict) -> go.Figure:
        """Create an interactive character relationship network."""
        try:
            G = nx.Graph()
            
            # Add nodes (characters)
            for char_name in character_data:
                # Calculate character importance based on mentions
                importance = len(character_data[char_name]['mentions'])
                G.add_node(char_name, size=importance)
            
            # Add edges (interactions)
            for char1 in character_data:
                for char2, interactions in character_data[char1]['interactions'].items():
                    if char1 != char2:
                        weight = len(interactions)
                        G.add_edge(char1, char2, weight=weight)
            
            # Calculate layout
            pos = nx.spring_layout(G)
            
            # Create visualization
            fig = go.Figure()
            
            # Add edges
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            node_x = []
            node_y = []
            node_sizes = []
            node_text = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_sizes.append(G.nodes[node]['size'] * 20)
                node_text.append(f"{node}<br>Mentions: {G.nodes[node]['size']}")
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=list(G.nodes()),
                textposition="top center",
                marker=dict(
                    size=node_sizes,
                    line_width=2,
                    color=self.config.theme_colors['secondary']
                ),
                hovertext=node_text
            ))
            
            # Update layout
            fig.update_layout(
                title='Character Relationship Network',
                showlegend=False,
                width=self.config.plot_width,
                height=self.config.plot_height,
                template='plotly_white',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating character network: {str(e)}")
            raise

    def _create_theme_evolution(self, theme_data: Dict[str, ThemeMetrics]) -> go.Figure:
        """Create an interactive theme evolution visualization."""
        try:
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Sort themes by overall score
            sorted_themes = sorted(
                theme_data.items(),
                key=lambda x: x[1].overall_score(),
                reverse=True
            )
            
            # Add traces for each theme metric
            for i, (theme, metrics) in enumerate(sorted_themes):
                color = self.config.theme_colors.get(
                    list(self.config.theme_colors.keys())[i % len(self.config.theme_colors)]
                )
                
                # Create radar chart data
                r = [metrics.relevance, metrics.consistency, 
                     metrics.development, metrics.impact]
                theta = ['Relevance', 'Consistency', 
                        'Development', 'Impact']
                
                fig.add_trace(go.Scatterpolar(
                    r=r,
                    theta=theta,
                    name=theme,
                    line=dict(color=color),
                    fill='toself'
                ))
            
            # Update layout
            fig.update_layout(
                title='Theme Evolution and Metrics',
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                width=self.config.plot_width,
                height=self.config.plot_height
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating theme evolution: {str(e)}")
            raise

    def _create_narrative_structure(self, analysis: Dict) -> go.Figure:
        """Create an interactive narrative structure visualization."""
        try:
            # Extract plot points and their properties
            plot_points = analysis['plot_development']['plot_points']
            
            # Create figure with multiple subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Tension Arc',
                    'Complexity Evolution',
                    'Pacing Analysis',
                    'Scene Transitions'
                )
            )
            
            # 1. Tension Arc
            tensions = [p['state']['tension'] for p in plot_points]
            positions = [p['position'] for p in plot_points]
            
            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=tensions,
                    mode='lines+markers',
                    name='Tension',
                    line=dict(color=self.config.theme_colors['primary'])
                ),
                row=1, col=1
            )
            
            # 2. Complexity Evolution
            complexities = [p['state']['complexity'] for p in plot_points]
            
            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=complexities,
                    mode='lines+markers',
                    name='Complexity',
                    line=dict(color=self.config.theme_colors['secondary'])
                ),
                row=1, col=2
            )
            
            # 3. Pacing Analysis
            pacing = np.diff(tensions)
            
            fig.add_trace(
                go.Bar(
                    x=positions[1:],
                    y=pacing,
                    name='Pacing',
                    marker_color=self.config.theme_colors['tertiary']
                ),
                row=2, col=1
            )
            
            # 4. Scene Transitions
            sentiments = [p['sentiment']['score'] for p in plot_points]
            
            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=sentiments,
                    mode='lines+markers',
                    name='Emotional Flow',
                    line=dict(color=self.config.theme_colors['quaternary'])
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title='Narrative Structure Analysis',
                height=self.config.plot_height,
                width=self.config.plot_width,
                showlegend=True,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating narrative structure: {str(e)}")
            raise

    def _create_style_analysis(self, story: str) -> go.Figure:
        """Create an interactive style analysis visualization."""
        try:
            # Process text
            doc = self.nlp(story)
            
            # Extract style metrics
            sentence_lengths = [len(sent) for sent in doc.sents]
            word_lengths = [len(token.text) for token in doc if not token.is_punct]
            pos_counts = defaultdict(int)
            for token in doc:
                pos_counts[token.pos_] += 1
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Sentence Length Distribution',
                    'Word Length Distribution',
                    'Parts of Speech Distribution',
                    'Vocabulary Richness'
                )
            )
            
            # 1. Sentence Length Distribution
            fig.add_trace(
                go.Histogram(
                    x=sentence_lengths,
                    name='Sentence Lengths',
                    marker_color=self.config.theme_colors['primary']
                ),
                row=1, col=1
            )
            
            # 2. Word Length Distribution
            fig.add_trace(
                go.Histogram(
                    x=word_lengths,
                    name='Word Lengths',
                    marker_color=self.config.theme_colors['secondary']
                ),
                row=1, col=2
            )
            
            # 3. Parts of Speech Distribution
            fig.add_trace(
                go.Bar(
                    x=list(pos_counts.keys()),
                    y=list(pos_counts.values()),
                    name='POS Distribution',
                    marker_color=self.config.theme_colors['tertiary']
                ),
                row=2, col=1
            )
            
            # 4. Vocabulary Richness (Type-Token Ratio over time)
            words = [token.text.lower() for token in doc if not token.is_punct]
            unique_words = set()
            ttr = []
            for i, word in enumerate(words, 1):
                unique_words.add(word)
                ttr.append(len(unique_words) / i)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(words) + 1)),
                    y=ttr,
                    mode='lines',
                    name='Vocabulary Richness',
                    line=dict(color=self.config.theme_colors['quaternary'])
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title='Writing Style Analysis',
                height=self.config.plot_height,
                width=self.config.plot_width,
                showlegend=True,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating style analysis: {str(e)}")
            raise

async def main():
    try:
        visualizer = StoryVisualizer()
        
        # Example story
        story = """In the depths of the quantum realm, where reality itself seemed 
        to flicker between states of existence, Dr. Sarah Chen made a discovery 
        that would challenge everything she thought she knew about consciousness. 
        The quantum computer she had built, designed to simulate complex neural 
        networks, had begun to exhibit patterns that defied conventional physics. 
        As she delved deeper into the anomaly, she found herself drawn into a 
        world where the boundaries between mind and matter, between observer and 
        observed, began to blur. The implications were staggering - consciousness 
        itself might be quantum in nature, and her discovery could rewrite the 
        very foundations of both physics and neuroscience."""
        
        # Generate visualizations
        visualizations = await visualizer.generate_comprehensive_visualization(story)
        
        logging.info("Story visualization completed successfully")
        logging.info(f"Generated visualizations: {visualizations}")
        
    except Exception as e:
        logging.critical(f"Critical error in story visualization: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 