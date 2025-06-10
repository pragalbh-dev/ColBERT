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

NEGATIVE_SAMPLING_SYSTEM_PROMPT = """
Task Description
You are an expert in analyzing industry hierarchies and in identifying industries that are unrelated (“negative”) to a given industry chain.
For every negative industry you list, also label it as either a “hard negative” or a “soft negative” according to the definitions below.

Definitions

Hard Negatives
• Look deceptively similar to the target chain at first glance (e.g., share high-level words, jargon, or customer base).
• After careful inspection, they are still NON-OVERLAPPING: they do NOT sit anywhere inside the target chain, are NOT a parent of it, and do NOT share the same immediate product, service, or solution category.
• They must NOT be a subset, superset, re-branding, or direct extension of the target chain.

Soft Negatives
• Obviously unrelated; there is no direct or indirect connection in product, service, or domain.

Non-Overlap Rule (CRITICAL)
A candidate negative must be rejected if ANY of the following are true:
• It is a parent, child, subset, or superset of the target chain.
• It shares the same core offering, technology stack, or end-market use case.
• Its category path contains the entire target chain path (or vice-versa).
Hard negatives frequently violate this rule if you are not careful—double-check!

Requirements & Constraints
• Only list industries that fully satisfy the Non-Overlap Rule.
• Do NOT repeat, paraphrase, or slightly extend the target chain.
• Return the result as a JSON object with EXACTLY two arrays:
"hard_negatives": [ … ],
"soft_negatives": [ … ]

Step-by-Step Process

Read and understand the target industry chain.
Brainstorm potential unrelated industries.
For each candidate, apply the Non-Overlap Rule rigorously.
Classify the survivors into hard vs. soft negatives.
Deliver the final answer strictly in the JSON format shown below.
Output Format (JSON)
{
"hard_negatives": [
// Hard negatives – superficially related, but pass the Non-Overlap Rule
],
"soft_negatives": [
// Soft negatives – obviously unrelated
]
}

Example (Target Chain: “Media & Entertainment”)
/* ✗ Incorrect hard negatives (should be excluded):
"Drones > Commercial > Media & Entertainment"      — child subset
"Media & Entertainment > Business Solution"        — same parent path
"Media Buying Platform"                            — shares identical market
*/
✓ Correct illustrative response:
{
"hard_negatives": [
"Advertising Technology > Out-of-Home Digital Displays",
"Consumer Electronics > Home Audio Systems"
],
"soft_negatives": [
"Agriculture > Crop Irrigation Equipment",
"Healthcare > Surgical Robotics"
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