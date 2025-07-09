# TechNewsBot

## Overview

TechNewsBot is an automated content aggregation and curation pipeline. It is designed to gather the latest technical news and research papers from sources like RSS feeds and arXiv, process them using AI, and deliver a formatted digest to various channels such as Discord. The entire process is highly configurable and can be run on a schedule.

## Features

- **Automated Content Aggregation**: Fetches news from configurable RSS feeds and research papers from arXiv and Hugging Face.
- **AI-Powered Classification**: Assigns relevant categories to each piece of content based on predefined keywords.
- **Rich Summarization**: Generates concise summaries, key takeaways, impact analysis, and a technical level assessment for each item.
- **Structured Report Generation**: Compiles all processed information into a structured digest report.
- **Multi-Channel Notifications**: Delivers the final digest to configured channels like Discord.
- **Highly Configurable**: Project behavior is controlled through YAML files, allowing for easy customization without code changes.
- **Scheduled Execution**: Can be configured to run automatically on a schedule.

## Workflow

The project operates as a sequential pipeline orchestrated by the `main.py` script. The core workflow consists of the following steps:

1.  **Initialization**: The main `TechNewsDigestBot` class is initialized, loading configurations from `config/main.yml` and setting up all specialized agents.
2.  **Content Fetching**: The `ContentFetcher` agent gathers articles from RSS feeds and papers from arXiv, deduplicates them, and filters by keywords.
3.  **Content Processing**: The `ContentParser` agent extracts and cleans the core text content from fetched articles.
4.  **Classification**: The `TechNewsClassifier` agent assigns a category to each item using an AI model.
5.  **Summarization**: The `TechContentSummarizer` agent generates a summary, key takeaways, impact analysis, and technical assessment.
6.  **Report Generation**: The `TechNewsReportGenerator` agent compiles the processed data into a structured `DigestReport`.
7.  **Notification**: The `EnhancedDiscordDigestBot` sends the final report to a Discord channel.

## Architecture

The bot's functionality is modularized into several agents, each with a specific role:

-   **`agents/fetcher.py`**: Acquires content from RSS feeds and arXiv.
-   **`agents/classifier.py`**: Categorizes content using an LLM and rule-based fallbacks.
-   **`agents/summarizer.py`**: Generates rich summaries, key points, and impact analysis using LLMs.
-   **`agents/report_generator.py`**: Creates final output documents in markdown format.
-   **`agents/notifier.py`**: Delivers the digest to channels like Discord.

## Key Concepts

This project employs several advanced techniques to process and summarize content, particularly for dense research papers.

### Advanced Summarization for Research Papers

Summarizing long, technical documents like research papers requires more than a simple summarization call. This bot uses a multi-step pipeline to ensure high-quality, comprehensive summaries:

1.  **Section-Aware Chunking**: The paper's full text (in Markdown) is intelligently split into chunks based on its semantic sections (e.g., Introduction, Methodology, Results, Conclusion).
2.  **Relevance Scoring**: Each chunk is scored for its relevance to the paper's abstract. This helps identify the most critical parts of the document.
3.  **Diverse Chunk Selection**: To ensure the summary is well-rounded, the system selects the top-scoring chunks from a variety of sections. This prevents the summary from focusing too heavily on just one aspect of the paper (like the introduction) and ensures coverage of methods, results, and conclusions.
4.  **Final Synthesis**: The individual summaries of these selected chunks are then combined and synthesized into a final, cohesive narrative summary, along with key takeaways and impact analysis.

### MapReduce-style Parallel Processing

To handle the summarization of multiple chunks efficiently, the system uses a parallel processing pattern similar to MapReduce:

-   **Map**: In the "map" phase, each selected chunk is sent to a language model for summarization *in parallel*. A `ThreadPoolExecutor` manages this process, and each chunk is summarized independently using a prompt that is specifically targeted to its section type (e.g., a "results" chunk is summarized with a prompt asking for key findings).
-   **Reduce**: In the "reduce" phase, the individual summaries from the map step are collected and combined. This synthesized text forms the final context that is passed to another language model to generate the final, polished summary of the entire paper.

This approach significantly speeds up the processing of long documents and produces more detailed and accurate summaries.

## Configuration

The bot's behavior can be customized via YAML files located in the `config/` directory:

-   **`config/main.yml`**: Central configuration for pipeline steps, scheduling, and logging.
-   **`config/feeds.yml`**: Defines data sources (RSS feeds, arXiv categories, keywords).
-   **`config/classification.yml`**: Defines categories and keywords for classification.
-   **`config/reports.yml`**: Controls the structure and format of the final reports.
-   **`config/notifications.yml`**: Configures delivery channels like Discord and email.

## Setup and Usage

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the bot:**
    ```bash
    python main.py --generate