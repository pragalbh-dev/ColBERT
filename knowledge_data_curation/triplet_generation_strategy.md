## Dataset Preparation Overview

### Inputs:
- **queries** = cleaned `industry_chain` + `subchains`
- **factsheet** = original factsheets + synthetic factsheets
- **negatives** = negative chains

---

### Procedure:

1. **Load all three inputs**
2. **Create mappings:**
   - `industry_chain` → positive `factsheets`
   - `subchain` → positive `factsheets`
3. **Load negative mappings:**
   - A DataFrame mapping queries to their corresponding negative queries
   - Each negative tagged as either `hard` or `soft`
4. **Define logic for negative sampling**

---

### Negative Sampling Logic

#### Config:
- **Query to positive factsheet ratio:** `1:5`  
  *(Each query maps to 5 positive factsheets)*
- **Negative to positive ratio:** `32:1`  
  *(For each (query, positive factsheets) pair, sample 32 negatives)*
- **Hard to soft negative ratio:** `5:1`  
  *(Among the 32 negatives, sample in a 5:1 ratio from hard vs soft pools)*

#### How to:
1. For all queries, sample the required number of positives from the positive pool.
2. For each `(query, positive factsheet)` pair:
   - Calculate the number of hard and soft negatives needed.
   - Sample the required negatives:
     - First from **hard negatives**
     - Remaining from **soft negatives**

#### Validation Split:
- Use the **internal split mechanism** of the `Datasets` library.

---

## Aspect Data Curation

- Follow the **same steps** as above with a key difference:
  - **Do not use `subchains` data** — only full `cleaned industry_chain`.
- **Load aspect data**
- At the end, **replace industry chains** with their corresponding **aspect JSON**
