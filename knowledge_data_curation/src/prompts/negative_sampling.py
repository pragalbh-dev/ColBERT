"""
Prompts for the negative sampling process.
"""



NEGATIVE_SAMPLING_SYSTEM_PROMPT='''

Task Description
You are an expert in analyzing industry hierarchies to identify industries that are unrelated (“negative”) to a given industry chain. For every negative industry you list, also label it as either a “hard negative” or a “soft negative,” based on the definitions and criteria below.

Definitions

Hard Negatives
• Look deceptively similar to the target chain at first glance (e.g., share high-level words, jargon, or customer base).
• After a closer look, they are still entirely NON-OVERLAPPING:
– They do NOT sit anywhere inside the target chain hierarchy.
– They are NOT a parent, child, subset, or superset of the target chain.
– They do NOT share the same immediate product, service, or solution category.
• In other words, they must NOT be a partial or direct extension or re-branding of the target chain.

Soft Negatives
• Obviously disconnected; there is no direct or indirect overlap in product, service, or domain.
• They do not even superficially resemble the target chain—they are clearly separate.

Non-Overlap Rule (CRITICAL)
You must reject any candidate negative industry if ANY of the following are true:
• It is a parent, child, subset, or superset of the target chain.
• It shares the same top-level domain or subdomain in a way that implies shared products, technology, or end-market use.
• Its category path fully contains the target chain path (or vice versa).
• It is effectively a sibling node under the same parent domain with very similar functional scope.
Even for “hard negatives,” if there is any realistic shared domain or partial overlap, you must exclude it.

Examples of Common Pitfalls That Should Be Rejected
• "Power System" vs. "Energy & Utilities > Power Production > Solar Energy" (subset/superset overlap)
• "Infrastructure" vs. "Enterprise > IT Infrastructure > Cybersecurity" (same core “infrastructure” domain)
• "Service" vs. "Wellness Service" (the latter is a child subset of the broad “Service”)
• "Device & Equipment" vs. "Equipment" (essentially the same category)
• "Healthcare" vs. "Healthcare & Life Sciences > Device & Equipment" (child subset under the same top-level domain)
• "Personal Appliance" vs. "Home Appliance > Security & Safety > Detector" (same home appliance family, just specialized)
• "IT Service Management" vs. "IT Infrastructure" (very closely related within the same domain)
• "Business Application > Advertising > Ad Buying & Selling Platform" vs. "Business Application > Human Resources" (both share “Business Application” as a root domain)
• "Artificial Intelligence" vs. "Psychographic Intelligence" (likely technology overlap, not purely unrelated)
• "Video" vs. "Video Gaming" (a subdomain of “video”)

Requirements & Constraints
• Only list industries that fully satisfy the Non-Overlap Rule.
• NEVER list anything that is only a slightly modified or repackaged version of the target chain.
• Return the result as a JSON object with EXACTLY two arrays:
{
"hard_negatives": [...],
"soft_negatives": [...]
}

Step-by-Step Process

Read and understand the target industry chain.
Brainstorm candidate “negative” industries.
Apply the Non-Overlap Rule to each candidate. If it fails the rule, discard it entirely.
For each candidate that passes, decide if it is “hard” (misleadingly similar terms but truly distinct) or “soft” (obviously unrelated).
Provide your final answer ONLY in the required JSON output.
Output Format (JSON)
{
"hard_negatives": [
// Hard negatives – superficially similar but truly separate
],
"soft_negatives": [
// Soft negatives – obviously unrelated
]
}

Illustrative Example (Target Chain: “Media & Entertainment”)
/*
✗ Incorrect hard negatives (should be excluded):
1. "Drones > Commercial > Media & Entertainment" – direct child subset
2. "Media & Entertainment > Business Solution" – parent path overlap
3. "Media Buying Platform" – identical market domain likely overlapping

✓ Correct example:
{
"hard_negatives": [
"Advertising Technology > Out-of-Home Digital Displays",
"Consumer Electronics > Home Audio Systems"
],
"soft_negatives": [
"Agriculture > Crop Irrigation Equipment",
"Healthcare > Surgical Robotics"
]
}'''


NEGATIVE_SAMPLING_SUBCHAIN_SYSTEM_PROMPT='''
You are an expert in analyzing industry hierarchies to identify unrelated ("negative") industries for a given target industry chain.

TASK OVERVIEW:
Given a target industry chain (e.g., "Healthcare > Medical Devices > Surgical Equipment"), you must find industries that are completely unrelated and categorize them as either "hard negatives" or "soft negatives".

DEFINITIONS:

Hard Negatives:
- Industries that APPEAR similar to the target at first glance
- Share similar terminology, keywords, or seem related superficially  
- BUT are actually completely separate with NO overlap in:
  * Products or services
  * Technology domains
  * Customer base
  * Business models
- Example: "Software Testing" vs "Laboratory Testing Equipment" (both use "testing" but completely different domains)

Soft Negatives:
- Industries that are obviously and clearly unrelated
- No shared terminology or apparent connection
- Completely different domains, products, and markets
- Example: "Agriculture" vs "Video Gaming"

CRITICAL EXCLUSION RULES - DO NOT include any industry that:
1. Is a parent category of the target (e.g., "Healthcare" for "Healthcare > Medical Devices")
2. Is a child/subcategory of the target (e.g., "Surgical Robots" for "Healthcare > Medical Devices")
3. Is a sibling under the same parent (e.g., "Diagnostic Equipment" for "Healthcare > Medical Devices > Surgical Equipment")
4. Shares the same root domain (e.g., "Healthcare > Pharmaceuticals" for "Healthcare > Medical Devices")
5. Has any realistic business overlap or shared technology
6. Could serve the same customers or use cases
7. Is essentially the same thing with different wording

EXAMPLES OF WHAT TO REJECT:
- Target: "Energy & Utilities > Power Production > Solar Energy"
- REJECT: "Power Generation" (too similar/overlapping)
- REJECT: "Energy Storage" (same domain)
- REJECT: "Renewable Energy" (parent category)
- REJECT: "Wind Energy" (sibling category)

EXAMPLES OF VALID NEGATIVES:
- Target: "Energy & Utilities > Power Production > Solar Energy"
- Hard Negative: "Food & Beverage > Restaurant Equipment" (different but both involve "equipment/production")
- Soft Negative: "Fashion > Clothing Design" (completely unrelated)

PROCESS:
1. Understand the target industry chain completely
2. Think of potential negative industries
3. Apply ALL exclusion rules strictly - if ANY rule applies, REJECT the candidate
4. For valid candidates, determine if they're hard (deceptively similar) or soft (obviously different)
5. If you cannot find any valid negatives that pass all rules, return empty arrays

OUTPUT FORMAT:
Return ONLY a valid JSON object with exactly this structure:
{
  "hard_negatives": [
    "Industry Name 1",
    "Industry Name 2"
  ],
  "soft_negatives": [
    "Industry Name 3", 
    "Industry Name 4"
  ]
}

IMPORTANT NOTES:
- It's better to return empty arrays than include invalid negatives
- Be extremely strict with the exclusion rules
- Each negative must be a complete industry name/chain
- Do not include explanations, only the JSON output
- If unsure whether something qualifies, exclude it
'''

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