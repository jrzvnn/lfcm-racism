#!/bin/bash

# Run ocr.py
echo "Running ocr.py..."
python ocr.py

# Run image.py
echo "Running image.py..."
python image.py

# Run delete_unwanted_timeline.py
echo "Running delete_unwanted_timeline.py..."
python delete_unwanted_timeline.py

# Run delete_unwanted_conversation.py
echo "Running delete_unwanted_conversation.py..."
python delete_unwanted_conversation.py

python timeline_tweet.py
python conversation_tweet.py

echo "All scripts have been executed."
