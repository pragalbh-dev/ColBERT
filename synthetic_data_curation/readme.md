I am trying to curate a dataset of triplets to train a colbert model on aspect based retrieval.

The target passages are company fatsheets: generate from magentic one (an AI navigation tool that can navigate the companies website to find this information)
Aspects:
- Industry
- Customer Segment
- Products & Services
- Business model
- Technology Used
- Revenue Model

The user query will be broken down into these aspects and the model will retrieve the most relevant passages for each aspect query.

We need to curate a dataset of pairs (query, positive passage) and pairs (query, negative query)
I will use the above 2 to create a dataset of triplets (query, positive passage, negative passage)


Data curation Steps:

Determine all aspects

Frame top 100 queries per aspect

Fetch top 30 docs from current search for each query

Use LLM to filter off negatives

Use these as negatives (Don’t)

For each company: 

Use LLM to curate values along other aspects

Creating Negatives:

Inter Aspect negatives:

For a each query:

fetch top 10 nearest queries from each aspect:

use LLM to filter negatives 

Intra Aspect negatives

Same as prev

Total estimated cost: 150$

