#!/usr/bin/env python3
"""
Topic modeling script using BERTopic on MongoDB data
"""
import sys
import traceback
import json
import pandas as pd
from collections import Counter
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient


def print_topic_info(topic_model, topics, texts, max_topics=10):
    """
    Print detailed information about the discovered topics
    """
    # Get topic information
    topic_info = topic_model.get_topic_info()
    print(f"\nTotal unique topics found: {len(topic_info)}")
    
    # Print most common topics
    topic_counts = Counter(topics)
    print("\nMost common topics:")
    common_topics = topic_counts.most_common(max_topics)
    for topic_id, count in common_topics:
        if topic_id == -1:
            print(f"  Topic {topic_id} (Outlier): {count} documents")
        else:
            words = topic_model.get_topic(topic_id)
            if words:
                words_str = ", ".join([word for word, _ in words[:5]])
                print(f"  Topic {topic_id}: {count} documents - Key terms: {words_str}")
            else:
                print(f"  Topic {topic_id}: {count} documents - No key terms available")
    
    # Print sample documents for the top topics
    print("\nSample documents for top topics:")
    for topic_id, _ in common_topics[:3]:
        if topic_id != -1:  # Skip outlier topic
            print(f"\nSample documents for Topic {topic_id}:")
            topic_docs = [text for i, text in enumerate(texts) if topics[i] == topic_id]
            for doc in topic_docs[:3]:
                print(f"  - {doc[:100]}..." if len(doc) > 100 else f"  - {doc}")


def run_topic_modeling():
    try:
        # Connect to MongoDB
        print("Connecting to MongoDB...")
        client = MongoClient('localhost', 27017)  # Adjust connection settings as needed
        db = client['samsung_social']  # Connect to the samsung_social database
        
        # Check connection
        try:
            client.admin.command('ping')
            print("MongoDB connection successful")
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            return
        
        # Load multilingual embedding model
        print("Loading embedding model...")
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        # Pull all comment texts
        print("Retrieving comments from database...")
        comment_query = {"text": {"$exists": True}}
        print(f"Comment query: {json.dumps(comment_query)}")
        
        # Count documents before filtering
        total_comment_count = db.comments.count_documents(comment_query)
        print(f"Total comments with 'text' field: {total_comment_count}")
        
        # Get all matching documents
        comment_docs = list(db.comments.find(comment_query))
        print(f"Retrieved {len(comment_docs)} comment documents")
        
        if not comment_docs:
            print("Warning: No comment documents found")
            comment_texts = []
        else:
            # Print sample of first few documents for verification
            print("Sample comment documents:")
            for i, doc in enumerate(comment_docs[:3]):
                doc_id = doc.get("_id", "unknown")
                text = doc.get("text", "")
                print(f"  Document {i+1} (ID: {doc_id}): text = '{text[:50]}{'...' if len(text) > 50 else ''}'")
                print(f"  Document {i+1} keys: {list(doc.keys())}")
            
            # Filter for non-empty text
            comment_texts = [doc["text"] for doc in comment_docs if doc.get("text") and doc.get("text").strip()]
            print(f"Found {len(comment_texts)} non-empty comments after filtering")
            print(f"Found {len(comment_texts)} non-empty comments")

        # Pull all indirect captions
        # Pull all indirect captions
        print("Retrieving indirect mentions from database...")
        mention_query = {"caption": {"$exists": True}}
        print(f"Mention query: {json.dumps(mention_query)}")
        
        # Count documents before filtering
        total_mention_count = db.indirect_mentions.count_documents(mention_query)
        print(f"Total indirect mentions with 'caption' field: {total_mention_count}")
        
        # Get all matching documents
        mention_docs = list(db.indirect_mentions.find(mention_query))
        print(f"Retrieved {len(mention_docs)} indirect mention documents")
        
        if not mention_docs:
            print("Warning: No indirect mention documents found")
            mention_texts = []
        else:
            # Print sample of first few documents for verification
            print("Sample indirect mention documents:")
            for i, doc in enumerate(mention_docs[:3]):
                doc_id = doc.get("_id", "unknown")
                caption = doc.get("caption", "")
                print(f"  Document {i+1} (ID: {doc_id}): caption = '{caption[:50]}{'...' if len(caption) > 50 else ''}'")
                print(f"  Document {i+1} keys: {list(doc.keys())}")
            
            # Filter for non-empty captions
            mention_texts = [doc["caption"] for doc in mention_docs if doc.get("caption") and doc.get("caption").strip()]
            print(f"Found {len(mention_texts)} non-empty captions after filtering")
        # Fit BERTopic model for comments
        if comment_texts:
            print("Fitting BERTopic on comments...")
            try:
                topic_model_comments = BERTopic(embedding_model=embedding_model, language="multilingual")
                topics_comments, probs_comments = topic_model_comments.fit_transform(comment_texts)
                
                # Log detailed topic information instead of visualization
                print_topic_info(topic_model_comments, topics_comments, comment_texts)
                
                # Save topic information to CSV for later analysis
                try:
                    topic_df = topic_model_comments.get_topic_info()
                    topic_df.to_csv("comment_topics.csv", index=False)
                    print("Topic information saved to comment_topics.csv")
                except Exception as e:
                    print(f"Warning: Could not save topic information to CSV: {e}")
                
                # Assign topic IDs back into MongoDB for comments
                print("Updating comment documents with topic IDs...")
                update_count = 0
                for i, doc in enumerate(comment_docs):
                    if i < len(topics_comments):  # Safety check
                        result = db.comments.update_one(
                            {"_id": doc["_id"]}, 
                            {"$set": {"topic_id": int(topics_comments[i])}}
                        )
                        if result.modified_count > 0:
                            update_count += 1
                print(f"Comment topic assignment complete: {update_count} documents updated")
            except Exception as e:
                print(f"Error processing comments: {e}")
                traceback.print_exc()
        
        # Fit BERTopic model for mentions
        if mention_texts:
            print("Fitting BERTopic on indirect mentions...")
            try:
                topic_model_mentions = BERTopic(embedding_model=embedding_model, language="multilingual")
                topics_mentions, probs_mentions = topic_model_mentions.fit_transform(mention_texts)
                
                # Log detailed topic information instead of visualization
                print_topic_info(topic_model_mentions, topics_mentions, mention_texts)
                
                # Save topic information to CSV for later analysis
                try:
                    topic_df = topic_model_mentions.get_topic_info()
                    topic_df.to_csv("mention_topics.csv", index=False)
                    print("Topic information saved to mention_topics.csv")
                except Exception as e:
                    print(f"Warning: Could not save topic information to CSV: {e}")
                
                # Assign topic IDs back into MongoDB for mentions
                print("Updating mention documents with topic IDs...")
                update_count = 0
                for i, doc in enumerate(mention_docs):
                    if i < len(topics_mentions):  # Safety check
                        result = db.indirect_mentions.update_one(
                            {"_id": doc["_id"]}, 
                            {"$set": {"topic_id": int(topics_mentions[i])}}
                        )
                        if result.modified_count > 0:
                            update_count += 1
                print(f"Mention topic assignment complete: {update_count} documents updated")
            except Exception as e:
                print(f"Error processing mentions: {e}")
                traceback.print_exc()
        
        # Process posts
        print("\nRetrieving posts from database...")
        post_query = {"caption": {"$exists": True}}
        print(f"Post query: {json.dumps(post_query)}")
        
        # Count documents before filtering
        total_post_count = db.posts.count_documents(post_query)
        print(f"Total posts with 'caption' field: {total_post_count}")
        
        # Get all matching documents
        post_docs = list(db.posts.find(post_query))
        print(f"Retrieved {len(post_docs)} post documents")
        
        if not post_docs:
            print("Warning: No post documents found")
            post_texts = []
        else:
            # Print sample of first few documents for verification
            print("Sample post documents:")
            for i, doc in enumerate(post_docs[:3]):
                doc_id = doc.get("_id", "unknown")
                caption = doc.get("caption", "")
                hashtags = doc.get("hashtags", [])
                print(f"  Document {i+1} (ID: {doc_id}): caption = '{caption[:50]}{'...' if len(caption) > 50 else ''}'")
                print(f"  Document {i+1} hashtags: {hashtags}")
                print(f"  Document {i+1} keys: {list(doc.keys())}")
            
            # Create enriched text using both caption and hashtags
            post_texts = []
            for doc in post_docs:
                if not doc.get("caption"):
                    continue
                
                caption = doc.get("caption", "").strip()
                if not caption:
                    continue
                
                # Add hashtags to the caption text if available
                hashtags = doc.get("hashtags", [])
                if hashtags:
                    hashtag_text = " ".join([f"#{tag}" for tag in hashtags])
                    enriched_text = f"{caption} {hashtag_text}"
                else:
                    enriched_text = caption
                
                post_texts.append(enriched_text)
            
            print(f"Found {len(post_texts)} non-empty posts after filtering")
            
        # Fit BERTopic model for posts
        if post_texts:
            print("Fitting BERTopic on posts...")
            try:
                topic_model_posts = BERTopic(embedding_model=embedding_model, language="multilingual")
                topics_posts, probs_posts = topic_model_posts.fit_transform(post_texts)
                
                # Log detailed topic information
                print_topic_info(topic_model_posts, topics_posts, post_texts)
                
                # Save topic information to CSV for later analysis
                try:
                    topic_df = topic_model_posts.get_topic_info()
                    topic_df.to_csv("post_topics.csv", index=False)
                    print("Topic information saved to post_topics.csv")
                except Exception as e:
                    print(f"Warning: Could not save topic information to CSV: {e}")
                
                # Assign topic IDs back into MongoDB for posts
                print("Updating post documents with topic IDs...")
                update_count = 0
                processed_count = 0
                
                # Create a mapping from processed texts back to original documents
                # Since we're using enriched text, we need to map back to original docs
                processed_docs = []
                for doc in post_docs:
                    if not doc.get("caption") or not doc.get("caption").strip():
                        continue
                    processed_docs.append(doc)
                    processed_count += 1
                
                if processed_count != len(topics_posts):
                    print(f"Warning: Number of processed posts ({processed_count}) doesn't match topics ({len(topics_posts)})")
                
                # Update documents with topic IDs
                for i, doc in enumerate(processed_docs):
                    if i < len(topics_posts):  # Safety check
                        result = db.posts.update_one(
                            {"_id": doc["_id"]}, 
                            {"$set": {"topic_id": int(topics_posts[i])}}
                        )
                        if result.modified_count > 0:
                            update_count += 1
                
                print(f"Post topic assignment complete: {update_count} documents updated")
            except Exception as e:
                print(f"Error processing posts: {e}")
                traceback.print_exc()
        
        print("Topic modeling process completed successfully")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    run_topic_modeling()

