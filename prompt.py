TEACHER_REWRITER_STAGE_1_PROMPT = """
Write a passage that answers the following query: {Query}
"""
   
TEACHER_REWRITER_STAGE_2_PROMPT = """
You are a search engine. Below are a query and its answer: 
query: {Query}
answer: {Answer}
In order to obtain information corresponding to this answer, please provide at least three rewritten queries. Do not answer the rewritten queries. Don't output any words other than the rewritten queries.  
"""

STUDENT_REWRITER_STAGE_2_PROMPT = """
You are a search engine. In order to obtain information for answering the query, please provide at least three rewritten queries. Do not answer the rewritten queries. Don't output any words other than the rewritten queries. The rewritten queries are split by '###'. Below are a query: 
query: {Query}
"""

# Example  
TEACHER_REWRITER_STAGE_1 = """ 
Write a passage that answers the following query: 
query: What does the legal term demur mean?
"""

ANSWER = """
The legal term demur is defined as a formal objection raised by a party in response to a pleading or motion filed by the opposing party. It indicates that the party does not agree with the claims or arguments presented in the pleading or motion and requests the court to dismiss or strike it... 
"""

TEACHER_REWRITER_STAGE_2 = """
You are a search engine. Below are a query and its answer: 
query: What does the legal term demur mean?
answer: {ANSWER}
In order to obtain information corresponding to this answer, please provide at least three rewritten queries. Do not answer the rewritten queries. Don't output any words other than the rewritten queries.  
"""

REWRITTEN_QUERIES = """
1. What is the purpose of a demurrer in the legal system?
2. How is a demurrer used in civil cases? 
3. Can a party challenge the legal sufficiency of a pleading through a demurrer?
"""

STUDENT_STAGE = """
You are a search engine. In order to obtain information for answering the query, please provide at least three rewritten queries. Do not answer the rewritten queries. Don't output any words other than the rewritten queries. The rewritten queries are split by '###'. Below are a query: 
query: What does GDP stand for?.
"""

REWRITTEN_QUERIES = """
1. What is the meaning of GDP?
2. How is GDP calculated?
3. Why is GDP an important economic indicator?
"""   