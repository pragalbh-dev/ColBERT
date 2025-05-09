import pandas as pd
def search(query_vec,cur):
    cur.execute(f"SET @query_vec=(\'{str(query_vec)}\'):>VECTOR(768):>BLOB;")
    query='''SELECT  company_id,descriptions, embedding <*> @query_vec AS emb_score
                FROM company_descriptions_fact_sheet 
                order by emb_score desc
                limit 100'''
    cur.execute(query)
    return pd.DataFrame(cur.fetchall())




