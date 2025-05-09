"""
Prompts for the chain cleaning process.
"""

CHAIN_CLEANING_SYSTEM_PROMPT = """
You are an expert at analyzing industry hierarchies and taxonomies. 
Your task is to clean up industry chains by removing unnecessary elements that don't contribute to the meaning when used as a search query.
"""

CHAIN_CLEANING_PROMPT = """
Industry chains consist of hierarchical terms separated by ' > ', where each parent term is followed by its child term.

Sometimes, some terms in these chains are unnecessary when the chain is used as a search query. 
These terms might be present for UI navigation or organization purposes but don't add semantic value to the query.

Examples of such unnecessary terms might include:
- Generic categorization terms like "Multiline" or "Other"
- UI navigation aids like "More" or "See all"
- Terms that are redundant given their context

Here's an industry chain to clean:
"{chain}"

Please remove any unnecessary terms that don't contribute to the meaning of this chain when used as a search query.
Return only the cleaned chain with the same format (terms separated by ' > '), without any explanation.
If all terms are necessary, return the original chain unchanged.
""" 