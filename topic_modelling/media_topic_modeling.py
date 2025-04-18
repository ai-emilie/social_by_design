#!/usr/bin/env python3
"""
Media Topic Modeling Script using CLIP and BERTopic
for multimodal topic modeling of images and videos from MongoDB
"""
import os
import sys
import json
import time
import shutil
import tempfile
import traceback
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path
from collections import Counter, defaultdict
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Deep learning imports
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer

# Import CLIP
try:
    import clip
except ImportError:
    print("Installing CLIP - this may take a moment...")
    os.system("pip install git+https://github.com/openai/CLIP.git")
    import clip

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("media_topic_modeling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("media_topic_modeling")

# Constants
TEMP_DIR = Path(tempfile.mkdtemp(prefix="media_topic_modeling_"))
MAX_WORKERS = 5  # Number of parallel downloads
TIMEOUT = 30  # Seconds to wait for media download
MAX_RETRIES = 3  # Number of download retries
BATCH_SIZE = 16  # Batch size for CLIP processing
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
FIELD_NAME = "media_topic_id"  # Field name to store in MongoDB

# Media topic modeling class
class MediaTopicModeling:
    def __init__(self, db_name="samsung_social", clip_model_name="ViT-B/32"):
        """Initialize the media topic modeling class"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize MongoDB connection
        self.client = MongoClient('localhost', 27017)
        self.db = self.client[db_name]
        
        # Load CLIP model
        logger.info(f"Loading CLIP model: {clip_model_name}")
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        
        # Initialize text embedding model
        self.text_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        # Create temp directory for media downloads
        self.media_dir = TEMP_DIR
        logger.info(f"Temporary directory for media: {self.media_dir}")
        
        # Initialize media information
        self.media_paths = {}
        self.post_docs = []
        self.image_features = []
        self.post_ids = []
        self.captions = []
        self.combined_features = []
    
    def __del__(self):
        """Clean up resources"""
        try:
            # Clean up temp directory
            if os.path.exists(self.media_dir):
                shutil.rmtree(self.media_dir)
                logger.info(f"Cleaned up temporary directory: {self.media_dir}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _is_valid_url(self, url):
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _download_media(self, doc):
        """Download media for a single document"""
        post_id = doc["_id"]
        media_url = doc.get("media_url")
        video_url = doc.get("video_url")
        
        if not media_url and not video_url:
            logger.warning(f"No media or video URL found for post: {post_id}")
            return post_id, None
        
        # Prefer media_url over video_url (most efficient)
        url = media_url if media_url else video_url
        
        if not self._is_valid_url(url):
            logger.warning(f"Invalid URL for post {post_id}: {url}")
            return post_id, None
        
        # Create filename based on post ID
        extension = os.path.splitext(urlparse(url).path)[1].lower()
        if not extension:
            extension = ".jpg"  # Default extension
        elif extension not in ALLOWED_IMAGE_EXTENSIONS:
            extension = ".jpg"  # Force acceptable extension
            
        filename = os.path.join(self.media_dir, f"{post_id}{extension}")
        
        # Try to download the media
        for attempt in range(MAX_RETRIES):
            try:
                headers = {"User-Agent": USER_AGENT}
                response = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)
                
                if response.status_code == 200:
                    with open(filename, 'wb') as f:
                        response.raw.decode_content = True
                        shutil.copyfileobj(response.raw, f)
                    
                    logger.info(f"Successfully downloaded media for post {post_id}")
                    return post_id, filename
                else:
                    logger.warning(f"Failed to download media for post {post_id}, status code: {response.status_code}")
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for post {post_id}: {e}")
            
            # Wait before retrying
            time.sleep(1)
        
        logger.error(f"All download attempts failed for post {post_id}")
        return post_id, None
    
    def download_all_media(self):
        """Download all media in parallel"""
        logger.info("Starting parallel media download")
        
        # Get all posts with media_url
        query = {"$or": [
            {"media_url": {"$exists": True, "$ne": ""}},
            {"video_url": {"$exists": True, "$ne": ""}}
        ]}
        self.post_docs = list(self.db.posts.find(query))
        
        if not self.post_docs:
            logger.warning("No posts with media found")
            return False
        
        logger.info(f"Found {len(self.post_docs)} posts with media to process")
        
        # Download media in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_post = {executor.submit(self._download_media, doc): doc for doc in self.post_docs}
            
            for future in as_completed(future_to_post):
                post_id, filename = future.result()
                if filename:
                    self.media_paths[post_id] = filename
        
        logger.info(f"Successfully downloaded {len(self.media_paths)} media files")
        
        if not self.media_paths:
            logger.warning("No media could be downloaded")
            return False
            
        return True
    
    def _extract_features_batch(self, image_batch, post_ids_batch):
        """Extract features from a batch of images using CLIP"""
        try:
            # Process images to CLIP format
            processed_images = torch.stack([self.clip_preprocess(img) for img in image_batch])
            
            # Move to appropriate device
            processed_images = processed_images.to(self.device)
            
            # Extract features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(processed_images)
                
            # Convert to numpy and normalize
            image_features = image_features.cpu().numpy()
            image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
            
            # Return features with post IDs
            return list(zip(post_ids_batch, image_features))
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            traceback.print_exc()
            return []
    
    def extract_visual_features(self):
        """Extract visual features from downloaded media"""
        logger.info("Starting visual feature extraction")
        
        post_ids_with_features = []
        features_list = []
        post_texts = []
        
        # Process in batches
        current_batch = []
        current_ids = []
        
        for doc in self.post_docs:
            post_id = doc["_id"]
            
            if post_id not in self.media_paths:
                continue
                
            # Try to open and process the image
            try:
                media_path = self.media_paths[post_id]
                img = Image.open(media_path).convert("RGB")
                
                # Store for batch processing
                current_batch.append(img)
                current_ids.append(post_id)
                
                # Get caption for later use
                caption = doc.get("caption", "")
                hashtags = doc.get("hashtags", [])
                hashtag_text = " ".join([f"#{tag}" for tag in hashtags]) if hashtags else ""
                
                # Combine caption and hashtags
                full_text = f"{caption} {hashtag_text}".strip()
                post_texts.append(full_text)
                
                # Process batch when it reaches the desired size
                if len(current_batch) >= BATCH_SIZE:
                    batch_results = self._extract_features_batch(current_batch, current_ids)
                    
                    for batch_id, feature in batch_results:
                        post_ids_with_features.append(batch_id)
                        features_list.append(feature)
                    
                    # Reset batch
                    current_batch = []
                    current_ids = []
                    
            except Exception as e:
                logger.error(f"Error processing image for post {post_id}: {e}")
        
        # Process remaining items in the last batch
        if current_batch:
            batch_results = self._extract_features_batch(current_batch, current_ids)
            
            for batch_id, feature in batch_results:
                post_ids_with_features.append(batch_id)
                features_list.append(feature)
        
        # Set class variables for later use
        self.post_ids = post_ids_with_features
        self.image_features = features_list
        self.captions = post_texts[:len(post_ids_with_features)]  # Match the length
        
        logger.info(f"Extracted visual features for {len(self.post_ids)} posts")
        
        return len(self.post_ids) > 0
    
    def combine_features(self):
        """Combine visual features with text features"""
        logger.info("Combining visual and text features")
        
        if not self.post_ids or not self.image_features:
            logger.warning("No visual features to combine")
            return False
            
        # Extract text features using SentenceTransformer
        if not self.captions:
            logger.warning("No captions available for text features")
            return False
            
        try:
            text_features = self.text_model.encode(self.captions)
            
            # Ensure same dimensionality for proper combination
            if len(text_features) != len(self.image_features):
                logger.error(f"Feature length mismatch: text={len(text_features)}, image={len(self.image_features)}")
                return False
                
            # Normalize text features
            text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
            
            # Create combined features (equal weight)
            self.combined_features = []
            for i in range(len(self.post_ids)):
                combined = np.concatenate([self.image_features[i], text_features[i]])
                self.combined_features.append(combined)
                
            logger.info(f"Successfully combined features for {len(self.combined_features)} posts")
            return True
            
        except Exception as e:
            logger.error(f"Error combining features: {e}")
            traceback.print_exc()
            return False
    
    def perform_topic_modeling(self):
        """Perform topic modeling on combined features"""
        logger.info("Performing topic modeling on combined features")
        
        if not self.combined_features or not self.post_ids:
            logger.warning("No combined features for topic modeling")
            return None, None
            
        try:
            # Convert features to numpy array
            features_array = np.array(self.combined_features)
            
            # Create BERTopic model
            vectorizer_model = CountVectorizer(stop_words="english")
            topic_model = BERTopic(
                vectorizer_model=vectorizer_model,
                n_gram_range=(1, 2),
                min_topic_size=2,
                verbose=True
            )
            
            # Fit the model
            topics, probs = topic_model.fit_transform(self.captions, features_array)
            
            # Get topic info
            topic_info = topic_model.get_topic_info()
            logger.info(f"Generated {len(topic_info)} topics")
            
            # Log topic information
            logger.info("Top topics:")
            topic_counts = Counter(topics)
            for topic_id, count in topic_counts.most_common(5):
                if topic_id == -1:
                    logger.info(f"  Topic {topic_id} (Outlier): {count} documents")
                else:
                    words = topic_model.get_topic(topic_id)
                    if words:
                        words_str = ", ".join([word for word, _ in words[:5]])
                        logger.info(f"  Topic {topic_id}: {count} documents - Key terms: {words_str}")
                    else:
                        logger.info(f"  Topic {topic_id}: {count} documents - No key terms available")
            
            # Save topic information to CSV
            try:
                topic_info.to_csv("media_topics.csv", index=False)
                logger.info("Topic information saved to media_topics.csv")
            except Exception as e:
                logger.warning(f"Could not save topic information to CSV: {e}")
                
            return topic_model, topics
            
        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            traceback.print_exc()
            return None, None
    
    def update_database(self, topics, topic_model):
        """Update MongoDB with topic assignments"""
        logger.info("Updating MongoDB with media topic assignments")
        
        if not topics or not self.post_ids:
            logger.warning("No topics or post IDs for database update")
            return 0
            
        update_count = 0
        for i, post_id in enumerate(self.post_ids):
            if i < len(topics):
                try:
                    # Update the document with topic ID and topic words if available
                    update_data = {FIELD_NAME: int(topics[i])}
                    
                    # Store topic words if available
                    if hasattr(topic_model, 'get_topic') and topics[i] != -1:
                        words = topic_model.get_topic(topics[i])
                        if words:
                            word_list = [word for word, _ in words[:10]]
                            update_data["media_topic_words"] = word_list
                    
                    # Update MongoDB
                    result = self.db.posts.update_one(
                        {"_id": post_id},
                        {"$set": update_data}
                    )
                    
                    if result.modified_count > 0:
                        update_count += 1
                    
                except Exception as e:
                    logger.error(f"Error updating document {post_id}: {e}")
        
        logger.info(f"Updated {update_count} documents with media topic IDs")
        return update_count
    
    def run_pipeline(self):
        """Run the complete media topic modeling pipeline"""
        logger.info("Starting media topic modeling pipeline")
        
        # Step 1: Download media files
        if not self.download_all_media():
            logger.error("Media download failed, aborting pipeline")
            return False
        
        # Step 2: Extract visual features
        if not self.extract_visual_features():
            logger.error("Visual feature extraction failed, aborting pipeline")
            return False
        
        # Step 3: Combine with text features
        if not self.combine_features():
            logger.error("Feature combination failed, aborting pipeline")
            return False
        
        # Step 4: Perform topic modeling
        topic_model, topics = self.perform_topic_modeling()
        if topic_model is None or topics is None:
            logger.error("Topic modeling failed, aborting pipeline")
            return False
        
        # Step 5: Update database
        update_count = self.update_database(topics, topic_model)
        if update_count == 0:
            logger.warning("No documents were updated with topic IDs")
        
        logger.info("Media topic modeling pipeline completed successfully")
        return True


def main():
    """Main entry point for the script"""
    try:
        # Print banner
        print("\n" + "="*80)
        print(" "*25 + "MEDIA TOPIC MODELING PIPELINE")
        print("="*80 + "\n")
        
        # Create and run the media topic modeling pipeline
        media_topic_modeling = MediaTopicModeling()
        success = media_topic_modeling.run_pipeline()
        
        if success:
            print("\n" + "="*80)
            print(" "*25 + "PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80 + "\n")
            print("Results have been saved to:")
            print("  - MongoDB (field: 'media_topic_id')")
            print("  - media_topics.csv (topic information)")
            print("  - media_topic_modeling.log (detailed log)")
            return 0
        else:
            print("\n" + "="*80)
            print(" "*25 + "PIPELINE FAILED")
            print("="*80 + "\n")
            print("Check 'media_topic_modeling.log' for details")
            return 1
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 130
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        traceback.print_exc()
        print(f"\nAn unexpected error occurred: {e}")
        print("Check 'media_topic_modeling.log' for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
