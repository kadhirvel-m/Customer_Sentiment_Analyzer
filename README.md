# AI-Powered Customer Sentiment Analysis
 
### Authors & Roles
- Kadhirvel M - Deep Learning Engineer
- Vanitha A - Frontend Development
- Danushiyaa M - Data Engineering
- Chiruhaas B V - REST API

## Overview  
This project is a **real-time AI-driven sentiment analysis system** designed to help businesses extract actionable insights from customer feedback. It features **drag-and-drop functionality** for seamless data ingestion, **multilingual support**, and an **interactive Business Intelligence (BI) chatbot** for querying sentiment insights in natural language.  

## Key Features  
- **Drag-and-Drop Upload** – Easily upload feedback datasets for instant analysis.  
- **Advanced Sentiment Analysis** – Utilizes **VADER, RoBERTa, and Mistral 7B** to classify sentiments as **positive, neutral, or negative**.  
- **Multilingual Support** – Processes feedback in multiple languages using contextual embeddings.  
- **Named Entity Recognition (NER) & Theme Extraction** – Identifies key topics and customer concerns.  
- **BI Chatbot with RAG** – Enables real-time querying of insights using **natural language processing (NLP)**.  

## Technical Stack  

### NLP Models & Libraries  
- **VADER** – Initial rule-based sentiment scoring.  
- **RoBERTa** – Deep contextual sentiment analysis.  
- **Mistral 7B** – Insight generation and factor extraction.  
- **spaCy & NLTK** – Text preprocessing & Named Entity Recognition (NER).  

### Backend & Deployment  
- **Flask** – API development.  
- **LangChain** – Connecting Mistral 7B for chatbot functionality.  
- **GPU/TPU Support** – Optimized for large-scale sentiment processing.  

### Frontend  
- **HTML, CSS, JavaScript** – Interactive UI for dataset upload and sentiment visualization.  

## System Workflow  
1. **Upload Feedback** – Users drag and drop customer feedback datasets.  
2. **Preprocessing** – Tokenization, stopword removal, and language detection.  
3. **Sentiment Analysis**  
   - **VADER** → Initial sentiment polarity scoring.  
   - **RoBERTa** → Deep sentiment classification.  
   - **Mistral 7B** → Insight extraction & sentiment factor clustering.  
4. **Multilingual NLP Processing** – Cross-lingual embeddings for language-specific sentiment detection.  
5. **BI Chatbot Integration** – Users can query insights interactively using natural language.  

## Deployment  
The system is **fully developed, optimized, and deployed**, making it ready for real-world business applications. It efficiently handles **high-frequency real-time sentiment analysis** across industries like **e-commerce, finance, healthcare, and customer service**.  

## Installation & Usage  
1. **Clone the Repository**  
   ```sh
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

