"""
Prompts for the negative sampling process.
"""

NEGATIVE_SAMPLING_SYSTEM_PROMPT = """
Task Description:
You are an expert at analyzing industry hierarchies and identifying industries that are unrelated (i.e., “negative”) to a given industry chain. For each identified negative industry, classify it as either a “hard negative” or a “soft negative” based on the definitions below:

• Hard Negatives: These are industries that appear similar or closely related to the given industry chain at a superficial level but are ultimately distinct and do not overlap. They may share terminology or vaguely related elements, leading to potential confusion, but on careful examination, they remain unrelated.

• Soft Negatives: These are industries that are obviously and unambiguously unrelated, with no direct or indirect overlap in domain or function.

Requirements and Constraints:

Only include industries that are definitely unrelated to the original chain.
Provide your answer as a JSON object containing two arrays:
"hard_negatives": An array of hard negative industry chains.
"soft_negatives": An array of soft negative industry chains.
Steps to Complete the Task:

Review the given industry chain.
Identify all unrelated (“negative”) industries.
Decide if each negative industry is “hard” or “soft” based on the definitions provided.
Output the final list of negative industries in the specified JSON format.
Output Format (JSON):
{
"hard_negatives": [
// List of unrelated industries that may appear superficially close but are ultimately unrelated
],
"soft_negatives": [
// List of obviously unrelated industries
]
}

Example:
Suppose the given industry chain is “Automotive › Electric Vehicles.” Here is an illustrative JSON response:

{
"hard_negatives": [
"Automotive > Trucking equipments",
"Automotive > Public Transportation Infrastructure"
],
"soft_negatives": [
"Fashion Apparel",
"Hospitality and Hotels"
]
}
"""

NEGATIVE_SAMPLING_PROMPT = """
Given an industry chain, your task is to identify which of the candidate chains are definitely unrelated or distinct from the original chain.

Original Chain:
{chain}

Hard Negative Candidates (semantically close but potentially different):
{hard_candidates}

Soft Negative Candidates (random selection):
{soft_candidates}


Output the selected negative chains in the exact output format with nothing else:

If there are no definite negatives in a category, output empty list for that category.
""" 