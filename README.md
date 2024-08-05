# PDF-Semantic-Search-Engine
A system that searches for any piece of text and returns which PDF document contains the most similar content. This involves reading PDFs, converting text to a searchable form, storing it efficiently, and then searching through it when needed.
## Organize Documents

We have a bunch of folders, and each folder represents a user. Inside each user's folder, there are several PDF documents.

### Search for a Document

Given a search phrase or query (some text), we need to find which PDF document it is most likely referring to.

### How to Do It

#### Extract Text from PDFs

We need to read the content of each PDF and break it down into manageable pieces of text (called "chunks").

#### Create Text Representations

We use a special tool (a transformer model) to turn each chunk of text into a numerical form (called "embeddings") that captures its meaning.

#### Store These Representations

We store all these numerical representations in a fast search database called FAISS. This makes it easy to quickly compare pieces of text.

#### Search with the Query

When we get a search query, we convert it into its numerical form using the same tool. We search in our FAISS database to find the pieces of text (chunks) that are most similar to our query.

#### Identify the Right Document

We look at the top matches (e.g., the top 10 most similar chunks) to see which PDF document they belong to. The document that appears the most among these top matches is likely the one we're looking for.

### Why We Do It This Way

- **Efficiency**: By converting text to numerical forms and using FAISS, we can quickly find matches even if we have a lot of documents.
- **Accuracy**: Using advanced models to understand the meaning of text helps us find the right documents even if the exact words aren't used.

### The Tools We Use

- **Transformers Model**: Converts text to embeddings (numerical representations).
- **FAISS**: A fast search database for finding similar embeddings.
- **LLMSherpa**: Helps us break down PDFs into text chunks.
