import logging
import sys
import os
import pdb  # Added for debugging
import time  # Added to ensure 'perform_computational_task' works

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Methods.story_manager import save_story, load_story  # Ensure both functions are defined
from Methods.Computational_Methods import process_story_segment
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    filename='recursive_story_processing.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def recursive_process(story, depth=0, max_depth=5, performance_data=None):
    if performance_data is None:
        performance_data = []

    logging.info(f"Processing depth {depth}.")

    if depth > max_depth:
        logging.info("Maximum recursion depth reached.")
        return story

    try:
        # Process the current segment of the story
        processed_story = process_story_segment(story, depth)
        logging.info(f"Processed story segment at depth {depth}.")

    except Exception as e:
        logging.error(f"Unexpected error at depth {depth}: {e}")
        pdb.set_trace()  # Start debugger on any other unexpected error
        raise

    # Record performance data (e.g., processing time)
    processing_time = perform_computational_task()
    performance_data.append(processing_time)
    logging.info(f"Recorded processing time: {processing_time:.4f} seconds.")

    # Recursive call for the next segment
    processed_story = recursive_process(processed_story, depth + 1, max_depth, performance_data)

    return processed_story

def perform_computational_task():
    # Placeholder for a computational task
    start_time = time.time()
    # Simulate processing time
    time.sleep(0.1)
    end_time = time.time()
    return end_time - start_time

def visualize_performance(performance_data):
    plt.figure(figsize=(10, 6))
    plt.plot(performance_data, marker='o')
    plt.title('Computational Performance Over Recursion Depth')
    plt.xlabel('Recursion Depth')
    plt.ylabel('Processing Time (s)')
    plt.grid(True)
    plt.savefig('performance_visualization.png')
    plt.close()
    logging.info("Performance visualization saved as 'performance_visualization.png'.")

def main():
    try:
        # Load the story from a file if needed
        # Uncomment below lines if 'load_story' is defined and you have an input story file
        # story = load_story('input_story.txt')
        # logging.info("Loaded input story.")

        # Placeholder story data
        story = "Once upon a time..."
        logging.info("Initialized placeholder story.")

        processed_story = recursive_process(story)
        save_story(processed_story, 'processed_story.txt')
        logging.info("Saved processed story.")

        # Assuming perform_computational_task records processing times
        performance_data = []  # This should be collected during processing
        visualize_performance(performance_data)
    except Exception as e:
        logging.critical(f"Unexpected critical error in main: {e}")
        pdb.set_trace()

if __name__ == "__main__":
    main()